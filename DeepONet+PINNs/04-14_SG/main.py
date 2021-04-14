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
from scipy.io import savemat
def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    batch_size = args.batch_size
    restart = args.restart

    loading_steps = 8
    num_cols = 122
    num_rows = 122
    
    constB = 92
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
    
    ## initialization
    model = opnn().to(device)
    model = model.float()    
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    hist_train = hist_init_tensor
    u_pred_pinn = np.zeros((14884,8))
    v_pred_pinn = np.zeros((14884,8))
    phi_pred_pinn = np.zeros((14884,8))
    hist_pred_pinn = np.zeros((14884,8))
    ## training
    def train(epoch, vDelta, x_tensor, hist_train, u_train, v_train, phi_train, istep):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_energy(vDelta, x_tensor, hist_train)
            print(f"step: {istep+1}, epoch: {epoch}, loss_pde: {loss_pde}")
            loss = loss_pde #+ loss_data
            train_loss[epoch, 0] = loss
            loss.backward()
            
            return loss
        
        optimizer.step(closure)
    

    
    ## Iterations
    for istep in range(0,loading_steps):
        if (istep == 0) :
            epochs = 30000
        else :
            epochs = args.epochs

        print('Start training...', flush=True)
        tic = time.time()
        vDelta = vDelta_tensor[istep:istep+1, 0]
        train_loss = np.zeros((epochs, 1))     
        for epoch in range(0, epochs):

            if epoch == 0:
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
            if epoch == epochs - 10000:
                optimizer = torch.optim.LBFGS(model.parameters(), lr = 0.0001)
            
            ## Training
            u_train = u_tensor[:,istep:istep+1]
            v_train = v_tensor[:,istep:istep+1]
            phi_train = phi_tensor[:,istep:istep+1]
            train(epoch, vDelta, x_tensor, hist_train, u_train, v_train, phi_train, istep)
    
        toc = time.time()
    
        print(f'total training time: {int((toc-tic)/60)} min', flush=True)
        trainLossFile = './loss_step'+ str(istep+1) +'.txt'
        np.savetxt(trainLossFile, train_loss)
        
        y_pred = model(vDelta, x_tensor)
        u_pred, v_pred, phi_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        hist_pred = model.history(u_pred, v_pred, x_tensor, hist_train).detach().unsqueeze(1)
        hist_train = hist_pred

        u_pred_np = to_numpy(u_pred).reshape(num_cols*num_rows, -1)
        v_pred_np = to_numpy(v_pred).reshape(num_cols*num_rows, -1)
        phi_pred_np = to_numpy(phi_pred).reshape(num_cols*num_rows, -1)
        hist_pred_np = to_numpy(hist_pred).reshape(num_cols*num_rows, -1)
        u_pred_pinn[:,istep:istep+1] = u_pred_np
        v_pred_pinn[:,istep:istep+1] = v_pred_np
        phi_pred_pinn[:,istep:istep+1] = phi_pred_np
        hist_pred_pinn[:,istep:istep+1] = hist_pred_np
        
        u_true = u[:,istep:istep+1]
        v_true = v[:,istep:istep+1]
        phi_true = phi[:,istep:istep+1]
        
        damage_pred = phi_pred_np.reshape(num_rows, num_cols)
        xDisp_pred = u_pred_np.reshape(num_rows, num_cols)
        yDisp_pred = v_pred_np.reshape(num_rows, num_cols)
        damage_true = phi_true.reshape(num_rows, num_cols)
        xDisp_true = u_true.reshape(num_rows, num_cols)
        yDisp_true = v_true.reshape(num_rows, num_cols)
        
        plotField(damage_pred, damage_true, xDisp_pred, xDisp_true, yDisp_pred, yDisp_true, istep, dump_train)
        
        ####################### Figure #######################
        x = np.linspace(1, epochs, epochs)
        
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
        fig.savefig('./loss_step'+str(istep+1)+'.png')

    savemat('PinnsData.mat', {'u':u_pred_pinn, 'v':v_pred_pinn, 'phi':phi_pred_pinn, 'hist':hist_pred_pinn})    
if __name__ == "__main__":
    main()
