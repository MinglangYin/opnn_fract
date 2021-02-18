"""
Author: Minglang Yin, minglang_yin@brown.edu
DeepONet for predicting displacement field.
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

    dim_br = [1, 4, 6, 8]
    dim_tr = [3, 20, 20, 20]
    lmd = 0 #1e-5

    loading_steps = 8
    num_cases = 9
    num_test = 2
    num_train = 7
    size_filter = 11
    if_gauss_filter = 1

    Param = {}
    Param['loading_steps'] = loading_steps
    Param['num_cases'] = num_cases
    Param['size_filter'] = size_filter
    Param['num_train'] = num_train
    Param['num_test'] = num_test
    Param['if_gauss_filter'] = if_gauss_filter

    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    img_train, x_train, y_train, img_test, x_test, y_test = prep_train_test(Param)

    ## scaling
    y_train = y_train*10
    y_test = y_test*10

    ##
    img_train_tensor = torch.tensor(img_train, dtype=torch.float).to(device)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)

    img_test_tensor = torch.tensor(img_test, dtype=torch.float).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    train_loss = np.zeros((args.epochs, 1))
    test_loss = np.zeros((args.epochs, 1))

    ## training
    def train(epoch, img, x, y):
        model.train()
        def closure():
            optimizer.zero_grad()
            y_out = model(img, x)

            ## L2 regularization
            # regularization_loss = 0
            # for param in model._flatten.parameters():
            #     regularization_loss += torch.sum(param**2)
            # for param in model._trunk.parameters():
            #     regularization_loss += torch.sum(param**2)
            # for param in model._out_layer.parameters():
            #     regularization_loss += torch.sum(param**2)

            loss = ((y_out - y)**2).mean() #+ lmd*regularization_loss
            train_loss[epoch, 0] = loss
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
        ## switch optimizers
        # if epoch == epochs - 100:
        #     optimizer = torch.optim.LBFGS(model.parameters())
        #     print("Training with LBFGS")
        
        ## Training
        train(epoch, img_train_tensor, x_train_tensor, y_train_tensor)
        ## Testing
        y_pred = to_numpy(model.forward(img_test_tensor, x_test_tensor))
        y_true = to_numpy(y_test_tensor)
        test_loss[epoch, 0] = ((y_pred - y_true)**2).mean()

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
    phi_pred = model.forward(img_test_tensor, x_test_tensor)
    phi_pred = to_numpy(phi_pred).reshape(num_test*loading_steps, 1, img_test.shape[2], img_test.shape[3])
    phi_true = y_test.reshape(num_test*loading_steps, 1, img_test.shape[2], img_test.shape[3])
    count = 0
    for i in range(0, num_test*loading_steps):
        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(11, 3))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        h = ax.imshow(phi_pred[i][0], cmap='coolwarm')
        ax.set_title('Testing case (Pred): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = ax.imshow(phi_true[i][0], cmap='coolwarm')
        ax.set_title('Testing case (Pred): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)
        
        ax = fig.add_subplot(gs[2])
        h = ax.imshow(abs(phi_true[i][0] - phi_pred[i][0]), cmap='coolwarm')
        ax.set_title('Testing case (Err): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        fig.savefig(dump_test + '/img_' + str(i) + '.png')
        plt.close()

    ## save train
    phi_pred = model.forward(img_train_tensor, x_train_tensor)
    phi_pred = to_numpy(phi_pred).reshape(num_train*loading_steps, 1, img_train.shape[2], img_train.shape[3])
    phi_true = y_train.reshape(num_train*loading_steps, 1, img_train.shape[2], img_train.shape[3])
    count = 0
    for i in range(0, num_train*loading_steps):
        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(11, 3))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        h = ax.imshow(phi_pred[i][0], cmap='coolwarm')
        ax.set_title('Testing case (Pred): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = ax.imshow(phi_true[i][0], cmap='coolwarm')
        ax.set_title('Testing case (Pred): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)
        
        ax = fig.add_subplot(gs[2])
        h = ax.imshow(abs(phi_true[i][0] - phi_pred[i][0]), cmap='coolwarm')
        ax.set_title('Testing case (Err): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        fig.savefig(dump_train + '/img_' + str(i) + '.png')
        plt.close()
    

    
if __name__ == "__main__":
    main()
