import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from typing import Any, Dict
# from model.decoder import GenDecoder
from model.t2mnet import T2MNet
from model.diff_utils import BetaScheduler,SigmaScheduler
from model.diff_utils import d_log_p_wrapped_normal
from model.data_utils import lattice_params_to_matrix_torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MAX_ATOMIC_NUM=100


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim=256,dropout=0.1):  #256,64
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TGDiffusion(nn.Module):
    def __init__(self,timesteps) -> None:
        super(TGDiffusion, self).__init__()
        self.decoder = T2MNet(hidden_dim=512, max_atoms=100, num_layers=6,
                              act_fn='silu', dis_emb='sin', num_freqs=128, edge_style='fc',
                              max_neighbors=20, cutoff=7, ln=True, ip=True, pred_type = True)
        self.text_projection = ProjectionHead(embedding_dim=768)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta_scheduler = BetaScheduler(timesteps=timesteps,scheduler_mode = 'cosine')
        self.sigma_scheduler = SigmaScheduler(timesteps=timesteps,sigma_begin = 0.005, sigma_end = 0.5)
        self.time_dim = 256
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = False
        self.keep_coords = False
        self.n_T = timesteps

        # discrete diffusion
        self.eps = 1e-6
        self.num_classes = 100
        q_onestep_mats = []
        for t in range(timesteps):
            beta = self.beta_scheduler.betas[t]
            diag = torch.full((self.num_classes,), 1. - beta, dtype=torch.float64)
            mat = torch.diag(diag, diagonal=0)
            # Add beta_t to the num_pixel_vals/2-th column for the absorbing state.
            mat[:, self.num_classes-1] += beta
            q_onestep_mats.append(mat)

        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)  # this will be used for q_posterior_logits
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"
        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        self.optim = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.6, patience=30, threshold=0.0001)

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        out = a[t - 1, x, :]
        return out

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps)
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classes,), print(f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}")

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.
        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0
        fact1 = self._at(self.q_one_step_transposed, t, x_t)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 1, x_0_logits, out)
        return bc

    def vb(self, dist1, dist2):
        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (torch.log_softmax(dist1 + self.eps, dim=-1)
                                                         - torch.log_softmax(dist2 + self.eps, dim=-1))
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def forward(self, batch):
        text_emb = self.text_projection(batch.text)
        text_emb = torch.squeeze(text_emb)

        # Diffusion Forward
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        t = times.repeat_interleave(batch.num_atoms, dim=0)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        atom_type = batch.atom_types-1
        noisy_atom_type = self.q_sample(atom_type, t,torch.rand((*atom_type.shape, self.num_classes),device=self.device))

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x, pred_type = self.decoder(text_emb, time_emb, noisy_atom_type, input_frac_coords, input_lattice,  batch.num_atoms, batch.batch)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        # vb loss on A
        true_q_posterior_logits = self.q_posterior_logits(atom_type, noisy_atom_type, t)
        pred_q_posterior_logits = self.q_posterior_logits(pred_type, noisy_atom_type, t)
        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        # ce loss on A
        pred_type = pred_type.flatten(start_dim=0, end_dim=-2)
        atom_type = atom_type.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(pred_type, atom_type)

        type_loss = 0.001 * vb_loss + ce_loss

        loss = loss_lattice + loss_coord + type_loss

        return loss, loss_lattice, loss_coord, type_loss

    @torch.no_grad()
    def sample(self, batch,batch_size, step_lr = 1e-5):
        text_emb = self.text_projection(batch.text)
        text_emb = torch.squeeze(text_emb)

        batch_size = batch.num_graphs
        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        atom_type_T = torch.randint(0, 99, (batch.num_nodes, 1)).to(device)
        atom_type_T = atom_type_T.squeeze(1)

        time_start = self.beta_scheduler.timesteps
        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : atom_type_T, #batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            a_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x, _ = self.decoder(text_emb,time_emb, a_t, x_t, l_t,  batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x, pred_type = self.decoder(text_emb,time_emb, a_t, x_t_minus_05, l_t_minus_05,  batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            t_a = torch.tensor([t] * a_t.shape[0], device=self.device)
            pred_q_posterior_logits = self.q_posterior_logits(pred_type, a_t, t_a)
            noise = torch.rand((*a_t.shape, self.num_classes), device=self.device)
            noise = torch.clip(noise, self.eps, 1.0)
            not_first_step = (t_a != 1).float().reshape((a_t.shape[0], *[1] * (a_t.dim())))
            gumbel_noise = -torch.log(-torch.log(noise))
            a_t_minus_1 = torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : a_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }
        traj[0]['atom_types'] = traj[0]['atom_types']+1
        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]),
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    @torch.no_grad()
    def sample_one(self, batch, batch_size, step_lr = 1e-5):
        batch_size = batch.num_graphs

        # CLS Embedding
        norm_sents = [normalize(s) for s in batch.text]
        encodings = tokenizer(norm_sents, return_tensors='pt', padding=True, truncation=True)
        if torch.cuda.is_available():
            encodings.to(device)
        with torch.no_grad():
            last_hidden_state = text_model(**encodings)[0]
        cls_emb = last_hidden_state[:, 0, :]
        text_emb = self.text_projection(cls_emb)

        l_T, x_T, a_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device), torch.rand([batch.num_nodes, 100]).to(self.device)

        # if self.keep_coords:
        #     x_T = batch.frac_coords
        # if self.keep_lattice:
        #     l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        atom_probs = F.softmax(a_T, dim=1)
        atom_type_T = torch.multinomial(atom_probs, num_samples=1).squeeze(1) + 1

        time_start = self.beta_scheduler.timesteps
        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : atom_type_T, #batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)
            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            alphas_cumprod_h = self.beta_scheduler.alphas_cumprod[t]
            c0_h = torch.sqrt(alphas_cumprod_h)
            c1_h = torch.sqrt(1. - alphas_cumprod_h)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            a_t = traj[t]['atom_types']

            a_one_hot = F.one_hot(a_t - 1, num_classes=100).float()

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x, pred_atom_probs = self.decoder(text_emb,time_emb, a_t, x_t, l_t, batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor
            rand_a = torch.randn_like(a_one_hot) if t > 1 else torch.zeros_like(a_one_hot)
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x, pred_atom_probs_05 = self.decoder(text_emb,time_emb, a_t, x_t_minus_05, l_t_minus_05, batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
            pred_atom_probs = c0_h * (pred_atom_probs_05 - c1_h * pred_atom_probs) + sigmas * rand_a
            a_t_minus_1 = torch.argmax(pred_atom_probs, dim=1) + 1


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : a_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]),
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack
    