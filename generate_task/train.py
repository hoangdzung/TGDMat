import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from config import config
from model.diffusion import TGDiffusion
from torch_geometric.data import DataLoader
from model.dataset import MaterialDataset
from utils import configure_save_path, plot_losses, save_model, load_model, delete_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_size(model):
    # Get total number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Each parameter is typically 4 bytes (float32)
    model_size_in_bytes = num_params * 4  # 4 bytes for float32

    # Convert to MB
    model_size_in_MB = model_size_in_bytes / (1024 ** 2)

    return model_size_in_MB, num_params

def train_epoch(dataloader, model):
    iters = len(dataloader)
    diff_losses, coord_losses, type_losses, lattice_losses = np.empty(iters), np.empty(iters), np.empty(iters),np.empty(iters)
    for i, batch in enumerate(dataloader):
        batch=batch.to(device)
        loss, loss_lattice, loss_coord, loss_type  = model(batch)
        model.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        model.optim.step()
        diff_losses[i] = loss.item()
        coord_losses[i] = loss_coord.item()
        type_losses[i] = loss_type.item()
        lattice_losses[i] = loss_lattice.item()
    return diff_losses.mean(), coord_losses.mean(), type_losses.mean(), lattice_losses.mean()

def validate(dataloader, model):
    iters = len(dataloader)
    diff_losses, coord_losses, type_losses, lattice_losses = np.empty(iters), np.empty(iters), np.empty(iters),np.empty(iters)

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        loss, loss_lattice, loss_coord, loss_type  = model(batch)
        diff_losses[i] = loss.item()
        coord_losses[i] = loss_coord.item()
        type_losses[i] = loss_type.item()
        lattice_losses[i] = loss_lattice.item()

    return diff_losses.mean(), coord_losses.mean(), type_losses.mean(), lattice_losses.mean()


def train(train_dataloader, val_dataloader, test_dataloader, model, config, dataset,epochs):
    basedir = configure_save_path(dataset)
    path = os.path.join(basedir, config.file_name_model)
    figpath = os.path.join(basedir, config.file_name_plot)
    min_loss = 1e8
    best_epoch = None

    if not os.path.exists(figpath):
        os.makedirs(figpath)
    out = open(figpath+"/out.txt", "w")

    best_model = model

    for epoch in tqdm(range(epochs)):
        t0 = time()
        loss, coord_loss, type_loss, lattice_loss = train_epoch(train_dataloader, model)
        val_loss, val_coord_loss, val_type_loss, val_lattice_loss = validate(val_dataloader, model)
        model.scheduler.step(loss)

        if loss < min_loss:
            save_model(model, f"{path}_{epoch:03d}.pt")
            if best_epoch is not None: delete_model(f"{path}_{best_epoch:03d}.pt")
            min_loss = loss
            best_epoch = epoch
            best_model = model
            best_str = "NEW BEST"
        else:
            best_str = f"best: {min_loss:.4f} at epoch " + str(best_epoch)
        if epoch % 10 ==0 :
            print(f"Epoch {epoch}/{epochs} Loss: {loss:.4f}, Coord Loss: {coord_loss:.4f}, "
                  f"Atom Type Loss: {type_loss:.4f}, Lattice Loss: {lattice_loss:.4f} [{time() - t0:.2f}s]",best_str)

            print(f"Validation Coord Loss: {val_coord_loss:.4f}, "
                  f"Atom Type Loss: {val_type_loss:.4f}, Lattice Loss: {val_lattice_loss:.4f}, [{time() - t0:.2f}s]")

            print("\n")

        results = 'Epoch :' + str(epoch) \
                  + ' Train Diff Loss : ' + str(round(loss,4)) \
                  + ' Train Coord Loss : ' + str(round(coord_loss,4)) \
                  + ' Train Type Loss : ' + str(round(type_loss,4)) \
                  + ' Train Lattice Loss : ' + str(round(lattice_loss,4)) \
                  + ' Valid Coord Loss : ' + str(round(val_coord_loss, 4)) \
                  + ' Valid Atom Type Loss : ' + str(round(val_type_loss, 4)) \
                  + ' Valid Lattice Loss : ' + str(round(val_lattice_loss, 4)) \
                  + ' Time : ' + str(int(time() - t0)) \

        out.writelines(results)
        out.writelines("\n")
        out.writelines("\n")

    _, test_coord_loss, test_type_loss, test_lattice_loss = validate(test_dataloader, model)
    print(f"Test Coord Loss: {test_coord_loss:.4f}, "
          f"Atom Type Loss: {test_type_loss:.4f}, Lattice Loss: {test_lattice_loss:.4f}, [{time() - t0:.2f}s]")

    test_results ='Test Coord Loss : ' + str(round(test_coord_loss, 4)) \
              + ' Test Type Loss : ' + str(round(test_type_loss, 4)) \
              + ' Test Lattice Loss : ' + str(round(test_lattice_loss, 4))
    out.writelines(test_results)
    out.writelines("\n")
    save_model(best_model, f"{path}_final.pt")
    print("All saved at ", path)
    return model,path



def main(args):
    torch.manual_seed(config.SEED)
    device = config.device
    if config.device is None or not torch.cuda.is_available():
        device = "cpu"
    model = TGDiffusion(args.timesteps).to(device)

    model_size_in_MB, num_params = get_model_size(model)
    print("Total Params: ",num_params)
    print("Model Size (in MB): ", round(model_size_in_MB,2))
    print("Prompt Type: ", args.prompt_type)

    root_path = "../data_text/"
    train_dataset = MaterialDataset(root_path, args.dataset, args.prompt_type, config.train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataset = MaterialDataset(root_path, args.dataset, args.prompt_type, config.eval_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_dataset = MaterialDataset(root_path, args.dataset, args.prompt_type, config.test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)



    train(train_dataloader, val_dataloader, test_dataloader, model, config, args.dataset, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', required=True, type=str, default='perov_5')
    parser.add_argument('--expt_date',  type=str)
    parser.add_argument('--expt_time',  type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--prompt_type', type=str, default='long') #long or short
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    main(args)
