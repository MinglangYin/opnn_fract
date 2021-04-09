"""
Author: Minglang Yin, minglang_yin@brown.edu, Somdatta Goswami, somdatta_goswami@brown.edu
DeepONet for predicting damage field using data from Phase-field Models.
"""
import os
import torch
import time
import numpy as np

from functions import *
#from mfnn import *
from pinn import *
from preproc import *
import scipy.io
from plotting import *
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

    loading_steps = 2
    num_cols = 122
    num_rows = 122
    
    constB = 1
    dataset = io.loadmat('dataset.mat')
    x = dataset['coord']
    hist_init = constB*dataset['hist_init'][:,0:1]
    phi = dataset['phi_train']
    u = dataset['u_train']
    v = dataset['v_train']
    stress_train = dataset['stress_train']

    ## create folders
    dump_train = './Predictions/Train/'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    vDelta_tensor = torch.tensor(stress_train, dtype=torch.float).to(device)
    x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=True).to(device)
    u_tensor = torch.tensor(u, dtype=torch.float, requires_grad=True).to(device)
    v_tensor = torch.tensor(v, dtype=torch.float, requires_grad=True).to(device)
    phi_tensor = torch.tensor(phi, dtype=torch.float, requires_grad=True).to(device)
    hist_init_tensor = torch.tensor(hist_init, dtype=torch.float).to(device)
    
    # fig = plt.figure(figsize=(5, 5))
    # plt.scatter(x[:,0], x[:,1], c=hist_init[:,0])
    # plt.savefig('hist.png')
    # exit()
    # print(f"hist_init: {hist_init.shape}")
    # print(f"hist_init_tensor: {hist_init_tensor.shape}")
    # scipy.io.savemat('hist.mat',{'hist': hist_init})
    # exit()
    ## initialization
    model = opnn().to(device)
    model = model.float()    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    ## training
    def train(epoch, vDelta, x, hist, u, v, phi):
        model.train()
        def closure():
            optimizer.zero_grad()
#            y_out = model.forward(vDelta, x)
#            u_out, v_out, phi_out = y_out[:, 0], y_out[:, 1], y_out[:, 2]
#            loss_data = (1e5*(u_out - u[:, 0])**2).mean() + (1e4*(v_out - v[:, 0])**2).mean() + ((phi_out - phi[:, 0])**2).mean()
#            loss_data = ((u_out - u[:, 0])**2).mean() + ((v_out - v[:, 0])**2).mean() + ((phi_out - phi[:, 0])**2).mean()
            loss_pde = model.loss_energy(vDelta, x, hist)

            print(f"step: {step+1}, epoch: {epoch}, loss_pde: {loss_pde}")
            loss = loss_pde #+ loss_data
            train_loss[epoch, 0] = loss
            loss.backward()
            return loss
        optimizer.step(closure)
    
    hist_pred = hist_init_tensor
    # print(f"hist_pred: {hist_pred.shape}")
    # exit(0)
    ## Iterations
    for step in range(0,loading_steps):
        
        print('Start training...', flush=True)
        tic = time.time()
        
        train_loss = np.zeros((args.epochs, 1))        
        vDelta = vDelta_tensor[step:step+1, 0]
        # print(f"vDelta: {vDelta}")
            
        for epoch in range(0, epochs):
            
            if epoch >= 30000:
               optimizer = torch.optim.LBFGS(model.parameters(), lr = 0.0001)
            
            ## Training
            u_true = u_tensor[:,step:step+1]
            v_true = v_tensor[:,step:step+1]
            phi_true = phi_tensor[:,step:step+1]
            train(epoch, vDelta, x_tensor, hist_pred, u_true, v_true, phi_true)
    
        toc = time.time()
    
        print(f'total training time: {int((toc-tic)/60)} min', flush=True)
        trainLossFile= './train_loss'+ str(step) +'.txt'
#        testLossFile= './test_loss'+ str(iter) +'.txt'
        np.savetxt(trainLossFile, train_loss)
#        np.savetxt(testLossFile, test_loss)
        
        y_pred = model(vDelta, x_tensor)
        u_pred, v_pred, phi_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        hist_pred = model.history(u_pred, v_pred, x_tensor, hist_pred)
        
        u_pred = to_numpy(u_pred).reshape(num_cols*num_rows, -1)
        v_pred = to_numpy(v_pred).reshape(num_cols*num_rows, -1)
        phi_pred = to_numpy(phi_pred).reshape(num_cols*num_rows, -1)
        u_true = u[:,step:step+1]
        v_true = v[:,step:step+1]
        phi_true = phi[:,step:step+1]
        
        damage_pred = phi_pred.reshape(num_rows, num_cols)
        damage_true = phi_true.reshape(num_rows, num_cols)
        xDisp_pred = u_pred.reshape(num_rows, num_cols)
        xDisp_true = u_true.reshape(num_rows, num_cols)
        yDisp_pred = v_pred.reshape(num_rows, num_cols)
        yDisp_true = v_true.reshape(num_rows, num_cols)
        
        plotField(damage_pred, damage_true, xDisp_pred, xDisp_true, yDisp_pred, yDisp_true, step, dump_train)
        
        ####################### Figure #######################
        num_epoch = train_loss.shape[0]
        x = np.linspace(1, num_epoch, num_epoch)
        
        ## plotting
        fig = plt.figure(constrained_layout=False, figsize=(6, 6))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, train_loss, color='blue', label='Training Loss')
#        ax.plot(x, test_loss, color='red', label='Test Loss', linestyle='dashed')
    
        ax.set_yscale('log')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.legend(loc='upper left')
        fig.savefig('./loss_his'+str(step+1)+'.png')
        
if __name__ == "__main__":
    main()
