import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import config
from model.dataset import *
from torch_geometric.data import Batch
from model.diffusion import TGDiffusion
from torch_geometric.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

recommand_step_lr = {
    'csp':{
        "perov_5": 5e-7,
        "carbon_24": 5e-6,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'csp_multi':{
        "perov_5": 5e-7,
        "carbon_24": 5e-7,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'gen':{
        "perov_5": 1e-6,
        "carbon_24": 1e-5,
        "mp_20": 5e-6
    },
}

def lattices_to_params_shape(lattices):
    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) / (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


def generation(loader, model, step_lr, batch_size, num_to_sample):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    for idx, batch in enumerate(loader):
        print(f'batch {idx+1} / {len(loader)}')
        for eval_idx in range(num_to_sample):
            if torch.cuda.is_available():
                batch.cuda()
            outputs, traj = model.sample(batch, batch_size, step_lr=step_lr)
            frac_coords.append(outputs['frac_coords'].detach().cpu())
            num_atoms.append(outputs['num_atoms'].detach().cpu())
            atom_types.append(outputs['atom_types'].detach().cpu())
            lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)

def display(loader, model, step_lr, batch_size, num_to_sample):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    mat_ids = []
    sam_ids = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample_one(batch, batch_size, step_lr=step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        mat_ids.extend(batch.mat_id)
        sam_ids.append(batch.sam_id)

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    sam_ids = torch.cat(sam_ids, dim=0)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms, mat_ids, sam_ids)

def main(args):
    model_path = Path(args.model_path)
    root_path = "../data_text/"

    if 'disp' not in args.tasks:
        test_dataset = MaterialDataset(root_path, args.dataset, args.prompt_type, config.test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    device = config.device
    if config.device is None or not torch.cuda.is_available():
        device = "cpu"

    chkpt_name = args.chkpt_path

    model = TGDiffusion(args.timesteps).to(device)

    chkpt = torch.load(chkpt_name, map_location=device)
    model.load_state_dict(chkpt["model"])

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi']['perov_5']
    if torch.cuda.is_available():
        model.to('cuda')

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()
        (frac_coords,  atom_types,_, lengths, angles,num_atoms) = generation(test_dataloader, model, step_lr, args.batch_size, args.num_to_samples)
        print('Generation Time :',time.time() - start_time)

        gen_out_name = 'eval_gen.pt'
        print('Saving Pt File..')
        # path = str(model_path) +'/' + str(args.dataset)
        path = os.path.join(model_path, args.dataset)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        PATH = path + '/' + gen_out_name
        torch.save({'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'time': time.time() - start_time
        }, PATH)
        print('Saving Pt File..Done')

    if 'disp' in args.tasks:
        disp_dataset = MaterialDispDataset(f"{args.dataset}_test.csv", args.num_to_samples)
        disp_dataloader = DataLoader(disp_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True) #config.batch_size
        print('Evaluate model on the display task.')
        start_time = time.time()
        (frac_coords,  atom_types,_, lengths, angles,num_atoms, mat_ids, sam_ids) = display(disp_dataloader, model, step_lr, args.batch_size, args.num_to_samples)
        print('Generation Time :',time.time() - start_time)

        gen_out_name = 'eval_disp.pt'
        print('Saving Pt File..')
        # path = str(model_path) +'/' + str(args.dataset)
        path = os.path.join(model_path, args.dataset)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        PATH = path + '/' + gen_out_name
        torch.save({'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'mat_ids': mat_ids,
            'sam_ids': sam_ids,
            'time': time.time() - start_time
        }, PATH)
        print('Saving Pt File..Done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['gen', 'disp'])
    parser.add_argument('--chkpt_path', required=True, type=str)
    parser.add_argument('--num_to_samples', default=1, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--dataset', required=True, type=str, default='perov_5')
    parser.add_argument('--prompt_type', type=str, default='long')  # long or short
    args = parser.parse_args()
    main(args)