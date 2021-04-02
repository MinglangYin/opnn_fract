"""
Author: Minglang Yin, minglang_yin@brown.edu, Somdatta Goswami, somdatta_goswami@brown.edu
DeepONet for predicting damage field using data from peridynamics.
"""
import os
import torch
import time
import numpy as np
import scipy.io as io

from functions import *
from mfnn import *
import scipy.io

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Dataset, DataLoader
matplotlib.use('Agg')

class adr_dataset(Dataset):
    def __init__(self, f_in, f_pde, x, y, f_bc, x_bc, device):
        self.f_in = torch.tensor(f_in, dtype=torch.float).to(device)
        self.f_pde = torch.tensor(f_pde, dtype=torch.float).to(device)
        self.x = torch.tensor(x, dtype=torch.float, requires_grad=True).to(device)
        self.y = torch.tensor(y, dtype=torch.float).to(device)
        self.f_bc = torch.tensor(f_bc, dtype=torch.float).to(device)
        self.x_bc = torch.tensor(x_bc, dtype=torch.float).to(device)

        self.len = f_in.shape[0]

    def __getitem__(self, index):
        return self.f_in[index], self.f_pde[index], self.x[index], self.y[index], self.f_bc[index], self.x_bc[index]
        
    def __len__(self):
        return self.len

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    batch_size = args.batch_size
    restart = args.restart

    dim_br = [100, 50, 50, 50, 50]
    dim_tr = [2, 50, 50, 50, 50]
    lmd = 0 #1e-5

    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    dataset = io.loadmat('dataset.mat')

    num_train = dataset['num_train'][0][0]
    num_test = dataset['num_test'][0][0]
    num_plot = dataset['num_plot'][0][0]

    dim_f = dataset['dim_f'][0][0]
    dim_in = dataset['dim_in'][0][0]
    dim_bc = dataset['dim_in'][0][0]

    xt_in = dataset['X_in_data']
    f_in = dataset['F_in_data']
    u_in = dataset['U_in_data']
    f_pde = dataset['F_pde_data']

    xt_bc = dataset['X_bc_data']
    f_bc = dataset['F_bc_data']
    u_bc = dataset['U_bc_data']

    xt_test = dataset['X_test_data']
    f_test = dataset['F_test_data']
    u_test = dataset['U_test_data']

    xt_plot = dataset['X_plot_data']
    f_plot = dataset['F_plot_data']
    u_plot = dataset['U_plot_data']

    train_dataset = adr_dataset(f_in, f_pde, xt_in, u_in, f_bc, xt_bc, device)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    ##
    # f_in_tensor = torch.tensor(f_in, dtype=torch.float).to(device)
    # xt_in_tensor = torch.tensor(xt_in, dtype=torch.float, requires_grad=True).to(device)
    # u_in_tensor = torch.tensor(u_in, dtype=torch.float).to(device)
    # f_pde_tensor = torch.tensor(f_pde, dtype=torch.float).to(device)

    f_bc_tensor = torch.tensor(f_bc, dtype=torch.float).to(device)
    xt_bc_tensor = torch.tensor(xt_bc, dtype=torch.float).to(device)
    u_bc_tensor = torch.tensor(u_bc, dtype=torch.float).to(device)
    
    f_test_tensor = torch.tensor(f_test, dtype=torch.float).to(device)
    xt_test_tensor = torch.tensor(xt_test, dtype=torch.float).to(device)

    f_plot_tensor = torch.tensor(f_plot, dtype=torch.float).to(device)
    xt_plot_tensor = torch.tensor(xt_plot, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    train_loss = np.zeros((args.epochs, 3))
    test_loss = np.zeros((args.epochs, 1))
    
    ## training
    def train(epoch, f, x, u, f_gov, f_bc, x_bc):
        model.train()

        def closure():
            optimizer.zero_grad()
            y_out = model.forward(f, x)
            loss_data = ((y_out - u)**2).mean()
            
            y_bc = model.forward(f_bc, x_bc)
            loss_bc = ((y_bc)**2).mean()
            loss_pde = model.loss_pde(f, x, f_gov)
            loss = loss_pde + loss_data + loss_bc

            print(f'Epoch: {epoch+1}, PDE Loss: {loss_pde:.6f}, Data Loss: {loss_data:.6f}, BC Loss: {loss_bc:.6f}', flush=True)

            train_loss[epoch, 0] = loss_pde
            train_loss[epoch, 1] = loss_data
            train_loss[epoch, 2] = loss_bc
            loss.backward()
            return loss
        optimizer.step(closure)
 
    ## Iterations
    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):
        
        if epoch == 20000:
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
        elif epoch == 30000:
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
        elif epoch == 39500:
            optimizer = torch.optim.LBFGS(model.parameters())
            print("Training with LBFGS")

        ## Training
        for i, (f_in_batch, f_pde_batch, x_batch, y_batch, f_bc_batch, xt_bc_batch) in enumerate(train_dataloader):
            print(f'mini-batch: {i}')
            # train(epoch, i, u_train_tensor, x_train_tensor, y_train_tensor)
            train(epoch, f_in_batch, x_batch, y_batch, f_pde_batch, f_bc_batch, xt_bc_batch)

        ## Testing
        u_pred = to_numpy(model(f_test_tensor, xt_test_tensor))
        test_loss[epoch, 0] = ((u_pred - u_test)**2).mean()
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)

        ## Save model
        if (epoch+1)%save_step == 0: 
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)

    toc = time.time()

    print(f'total training time: {int((toc-tic)/60)} min', flush=True)
    np.savetxt('./train_loss.txt', train_loss)
    np.savetxt('./test_loss.txt', test_loss)

#################################################################
    u_pred = model(f_plot_tensor, xt_plot_tensor)
    u_pred = to_numpy(u_pred).reshape(num_plot, -1)
    u_true = u_plot.reshape(num_plot, -1)
    # scipy.io.savemat('u_test.mat',{'f':f_test, 'x':x_test, 'u':u_pred})

    for i in range(0, num_plot):
        print(f"printing testing case: {i}")
        u_model = u_pred[i, :].reshape(100, 100)
        u_data = u_true[i, :].reshape(100, 100)

        fig = plt.figure(constrained_layout=False, figsize=(14, 3))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        h = ax.imshow(u_model)
        ax.set_title('Testing case (Pred): ' + str(i))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1])
        h = ax.imshow(u_data)
        ax.set_title('Testing case (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[2])
        h = ax.imshow(abs(u_data - u_model))
        ax.set_title('Testing case (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_test + 'img_' + str(i) + '.png')
        plt.tight_layout()
        plt.close()

    # ## save train
    # u_pred = model(f_train_tensor, x_train_tensor)
    # u_pred = to_numpy(u_pred).reshape(num_train, -1)
    # u_true = u_train.reshape(num_train, -1)    
    # # scipy.io.savemat('u_train.mat',{'f':f_train, 'x':x_train, 'u':u_pred})

    # for i in range(0, num_train):
    #     print(f"printing training case: {i}")
    #     u_model = u_pred[i, :].reshape(100, 100)
    #     u_data = u_true[i, :].reshape(100, 100)

    #     fig = plt.figure(constrained_layout=False, figsize=(14, 3))
    #     gs = fig.add_gridspec(1, 3)

    #     ax = fig.add_subplot(gs[0])
    #     h = ax.imshow(u_model)
    #     ax.set_title('training case (Pred): ' + str(i))
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(h, ax=ax, cax=cax)

    #     ax = fig.add_subplot(gs[1])
    #     h = ax.imshow(u_data)
    #     ax.set_title('training case (True): ' + str(i))
    #     # h.set_clim(vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(h, ax=ax, cax=cax)

    #     ax = fig.add_subplot(gs[2])
    #     h = ax.imshow(abs(u_data - u_model))
    #     ax.set_title('training case (Err): ' + str(i))
    #     # h.set_clim(vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(h, ax=ax, cax=cax)

    #     fig.savefig(dump_train + 'img_' + str(i) + '.png')
    #     plt.tight_layout()
    #     plt.close()

    ## training loss
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    
    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss, color='blue', label='Training Loss')
    ax.plot(x, test_loss, color='red', label='Test Loss', linestyle='dashed')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig('./loss_his.png')
    
if __name__ == "__main__":
    main()
