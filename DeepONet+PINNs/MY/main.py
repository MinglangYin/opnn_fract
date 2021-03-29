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
from preproc import *
import scipy.io

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    batch_size = args.batch_size
    restart = args.restart

    dim_br = [100, 50, 50, 50]
    dim_tr = [2, 50, 50, 50]
    lmd = 0 #1e-5

    num_cases = 50
    num_train = 40
    num_test = 10

    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    dataset = io.loadmat('dataset.mat')
    dim_f = dataset['dim_f'][0][0]
    dim = dataset['dim'][0][0]

    x_data = dataset['X_data']
    f_data = dataset['F_data']
    u_data = dataset['U_data']
    f_pde_data = dataset['F_PDE_data']

    x_train = x_data[:num_train*dim, :]
    f_train = f_data[:num_train*dim, :]
    u_train = u_data[:num_train*dim, :]
    f_gov_train = f_pde_data[:num_train*dim, :]

    x_test = x_data[num_train*dim:(num_train+num_test)*dim, :]
    f_test = f_data[num_train*dim:(num_train+num_test)*dim, :]
    u_test = u_data[num_train*dim:(num_train+num_test)*dim, :]
    f_gov_test = f_pde_data[num_train*dim:(num_train+num_test)*dim, :]

    ##
    f_train_tensor = torch.tensor(f_train, dtype=torch.float).to(device)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float, requires_grad=True,).to(device)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_gov_train_tensor = torch.tensor(f_gov_train, dtype=torch.float).to(device)
    
    f_test_tensor = torch.tensor(f_test, dtype=torch.float).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float, requires_grad=True,).to(device)
    u_test_tensor = torch.tensor(u_test, dtype=torch.float).to(device)
    f_gov_test_tensor = torch.tensor(f_gov_test, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    train_loss = np.zeros((args.epochs, 1))
    test_loss = np.zeros((args.epochs, 1))
    
    ## training
    def train(epoch, f, x, u, f_gov):
        model.train()

        def closure():
            optimizer.zero_grad()
            y_out = model.forward(f,x)
            loss_data = ((y_out - u)**2).mean()
            loss_pde = 0#model.loss_pde(f,x,f_gov)
            loss = loss_pde + loss_data

            train_loss[epoch, 0] = loss
            loss.backward()
            return loss
        optimizer.step(closure)
 
    ## Iterations
    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):
        
        ## Training
        train(epoch, f_train_tensor, x_train_tensor, u_train_tensor, f_gov_train_tensor)

        ## Testing
        u_pred = to_numpy(model(f_test_tensor, x_test_tensor))
        u_true = to_numpy(u_test_tensor)
        test_loss[epoch, 0] = ((u_pred - u_true)**2).mean()
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
    u_pred = model(f_test_tensor, x_test_tensor)
    u_pred = to_numpy(u_pred).reshape(num_test, -1)
    u_true = u_test.reshape(num_test, -1)
    scipy.io.savemat('u_test.mat',{'f':f_test, 'x':x_test, 'u':u_pred})

    ## figure (test) ##

    for i in range(0, num_test):
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

    ## save train
    u_pred = model(f_train_tensor, x_train_tensor)
    u_pred = to_numpy(u_pred).reshape(num_train, -1)
    u_true = u_train.reshape(num_train, -1)    
    scipy.io.savemat('u_train.mat',{'f':f_train, 'x':x_train, 'u':u_pred})

    for i in range(0, num_train):
        print(f"printing training case: {i}")
        u_model = u_pred[i, :].reshape(100, 100)
        u_data = u_true[i, :].reshape(100, 100)

        fig = plt.figure(constrained_layout=False, figsize=(14, 3))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        h = ax.imshow(u_model)
        ax.set_title('training case (Pred): ' + str(i))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1])
        h = ax.imshow(u_data)
        ax.set_title('training case (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[2])
        h = ax.imshow(abs(u_data - u_model))
        ax.set_title('training case (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_train + 'img_' + str(i) + '.png')
        plt.tight_layout()
        plt.close()
    
if __name__ == "__main__":
    main()
