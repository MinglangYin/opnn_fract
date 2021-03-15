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

import matplotlib
matplotlib.use('Agg')

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    save_step = args.save_step
    batch_size = args.batch_size
    restart = args.restart

    dim_br = [25, 100, 100, 100]
    dim_tr = [2, 100, 100, 100]
    lmd = 0 #1e-5

    num_cases = 1000
    num_test = 200
    num_train = 800
    #size_filter = 11
    if_gauss_filter = 1
    sigma_gauss = 1.5
    
    Param = {}
    Param['num_cases'] = num_cases
    #Param['size_filter'] = size_filter
    Param['num_train'] = num_train
    Param['num_test'] = num_test
    Param['if_gauss_filter'] = if_gauss_filter
    Param['sigma_gauss'] = sigma_gauss
    
    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    u_train, x_train, y_train, u_test, x_test, y_test = prep_train_test(Param)

    ## scaling
    u_test = u_test*100
    u_train = u_train*100

    y_test = y_test*10
    y_train = y_train*10

    ##
    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)

    u_test_tensor = torch.tensor(u_test, dtype=torch.float).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    train_loss = np.zeros((args.epochs, 3))
    test_loss = np.zeros((args.epochs, 3))

    # print(y_train_tensor.shape)
    # print(x_train_tensor.shape)
    # print(u_train_tensor.shape)
    #exit()

    ## training
    def train(epoch, u, x, y):
        model.train()
        def closure():
            optimizer.zero_grad()
            y_out = model(u, x)

            ## L2 regularization
            #regularization_loss = 0
            # for param in model._flatten.parameters():
            #     regularization_loss += torch.sum(param**2)
            #for param in model._trunk.parameters():
                #regularization_loss += torch.sum(param**2)
            #for param in model._out_layer.parameters():
                #regularization_loss += torch.sum(param**2)
            #if (epoch <= 15000):
            m_loss = ((y_out - y)**2).mean()
                #reg_loss = lmd*regularization_loss
                #loss = m_loss + reg_loss
            #else: 
                #m_loss = abs(y_out - y).mean()
            #reg_loss = lmd*regularization_loss
            loss = m_loss #+ reg_loss
            train_loss[epoch, 0] = loss
            #train_loss[epoch, 1] = m_loss
            #train_loss[epoch, 2] = reg_loss
            loss.backward()
            return loss
        optimizer.step(closure)

    ## restart
    if restart == 1:
        print('Restart!', flush=True)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Iterations
    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):
        
        print(f'{epoch}')
        ## switch optimizers
        # if epoch == epochs - 100:
        #     optimizer = torch.optim.LBFGS(model.parameters())
        #     print("Training with LBFGS")
        
        ## Training
        train(epoch, u_train_tensor, x_train_tensor, y_train_tensor)
        ## Testing
        
        
        y_pred = to_numpy(model.forward(u_test_tensor, x_test_tensor))
        y_true = to_numpy(y_test_tensor)
        #if (epoch <= 15000):
        test_loss[epoch, 0] = ((y_pred - y_true)**2).mean()
        #else: 
        #test_loss[epoch, 0] = abs(y_pred - y_true).mean()
        #print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}, Mean Loss: {train_loss[epoch, 1]:.6f}, Regularization Loss: {train_loss[epoch, 2]:.6f} Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)
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
    damage_pred = model.forward(u_test_tensor, x_test_tensor)
    damage_pred = to_numpy(damage_pred).reshape(num_test, -1)
    damage_true = y_test.reshape(num_test, -1)

    # aa = int(74*193/10)
#    aa = 19*97
#
#    xx_test = x_test[0:aa, 0].reshape(97, 19)
#    yy_test = x_test[0:aa, 1].reshape(97, 19)

    # print(xx_test[0, :])
    # print(yy_test[0, :])
    # exit()
    num_cols = 19
    num_rows = 97
    count = 0
    for i in range(0, num_test):
        damage_pred_print = damage_pred[i, :].reshape(num_rows, num_cols).transpose()
        damage_true_print = damage_true[i, :].reshape(num_rows, num_cols).transpose()

        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(6, 12))
        gs = fig.add_gridspec(3, 1)

        ax = fig.add_subplot(gs[0])
        h = ax.imshow(damage_pred_print)
        ax.set_title('Testing case (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = ax.imshow(damage_true_print)
        ax.set_title('Testing case (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)
        
        ax = fig.add_subplot(gs[2])
        h = ax.imshow(abs(damage_pred_print - damage_true_print))
        ax.set_title('Testing case (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        fig.savefig(dump_test + '/img_' + str(i) + '.png')
        plt.close()

    ## save train
    damage_pred = model.forward(u_train_tensor, x_train_tensor)
    damage_pred = to_numpy(damage_pred).reshape(num_train, -1)
    damage_true = y_train.reshape(num_train, -1)

    count = 0
    for i in range(0, num_test):
        damage_pred_print = damage_pred[i, :].reshape(num_rows, num_cols).transpose()
        damage_true_print = damage_true[i, :].reshape(num_rows, num_cols).transpose()

        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(6, 12))
        gs = fig.add_gridspec(3, 1)

        ax = fig.add_subplot(gs[0])
        h = ax.imshow(damage_pred_print)
        ax.set_title('Training case (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = ax.imshow(damage_true_print)
        ax.set_title('Training case (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)
        
        ax = fig.add_subplot(gs[2])
        h = ax.imshow(abs(damage_pred_print - damage_true_print))
        ax.set_title('Training case (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        fig.savefig(dump_train + '/img_' + str(i) + '.png')
        plt.close()
    

    
if __name__ == "__main__":
    main()
