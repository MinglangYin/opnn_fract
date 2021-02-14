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
from pvstrut_dataset import *

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

    dim_br = [450, 50, 50, 50]
    dim_tr = [3, 50, 50, 50]
    lmd = 1e-5

    num_train = 4
    num_test = 4
    loading_steps = 8
    id_test = 450*num_test*loading_steps
    id_train = 450*num_train*loading_steps

    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    dataset = io.loadmat('../dataset.mat')
    phi0 = dataset['phi_init_train']
    x = dataset['x_train']
    force = dataset['stress_train']*100
    x = np.concatenate((x, force), axis=1)
    y = dataset['y_train']*2

    ##
    x_train = x[0:id_train]
    y_train = y[0:id_train]
    phi0_train = phi0[0:id_train]

    x_test = x[id_train: id_train + id_test]
    y_test = y[id_train: id_train + id_test]
    phi0_test = phi0[id_train: id_train + id_test]

    ## plotting
    # for i in range(0, num_train*loading_steps):

    #     x_plt = x_train[i*450:(i+1)*450, 0]
    #     y_plt = x_train[i*450:(i+1)*450, 1]
    #     phi0_plt = phi0_train[i*450, :]
    #     phi_plt = y_train[i*450:(i+1)*450, 0]

    #     fig = plt.figure(constrained_layout=False, figsize=(7, 3))
        
    #     gs = fig.add_gridspec(1, 2)
    #     ax = fig.add_subplot(gs[0])
    #     h = ax.scatter(x_plt, y_plt, c = phi0_plt, cmap='coolwarm')
    #     ax.set_title('Testing case (init): ' + str(i))
    #     h.set_clim(vmin=0, vmax=1)
    #     fig.colorbar(h, ax=ax)

    #     ax = fig.add_subplot(gs[1])
    #     h = ax.scatter(x_plt, y_plt, c = phi_plt, cmap='coolwarm')
    #     ax.set_title('Testing case (true): ' + str(i))
    #     h.set_clim(vmin=0, vmax=1)
    #     fig.colorbar(h, ax=ax)

    #     fig.savefig(dump_train + '/img_' + str(i) + '.png')
    #     plt.close()

    # for i in range(0, num_test*loading_steps):

    #     x_plt = x_train[i*450:(i+1)*450, 0]
    #     y_plt = x_train[i*450:(i+1)*450, 1]
    #     phi0_plt = phi0_test[i*450, :]
    #     phi_plt = y_test[i*450:(i+1)*450, 0]

    #     fig = plt.figure(constrained_layout=False, figsize=(7, 3))
    #     gs = fig.add_gridspec(1, 2)
    #     ax = fig.add_subplot(gs[0])
    #     h = ax.scatter(x_plt, y_plt, c = phi0_plt, cmap='coolwarm')
    #     ax.set_title('Training case (init): ' + str(i))
    #     h.set_clim(vmin=0, vmax=1)
    #     fig.colorbar(h, ax=ax)
        
    #     ax = fig.add_subplot(gs[1])
    #     h = ax.scatter(x_plt, y_plt, c = phi_plt, cmap='coolwarm')
    #     ax.set_title('Testing case (true): ' + str(i))
    #     h.set_clim(vmin=0, vmax=1)
    #     fig.colorbar(h, ax=ax)

    #     fig.savefig(dump_test + '/img_' + str(i) + '.png')
    #     plt.close()

    ##
    phi0_train_tensor = torch.tensor(phi0_train, dtype=torch.float).to(device)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)

    phi0_test_tensor = torch.tensor(phi0_test, dtype=torch.float).to(device)
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
        train(epoch, phi0_train_tensor, x_train_tensor, y_train_tensor)
        ## Testing
        y_pred = to_numpy(model.forward(phi0_test_tensor, x_test_tensor))
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
    ## save test
    # y_pred = model.forward(phi0_test_tensor, x_test_tensor)
    # y_pred = to_numpy(y_pred)
    # count = 0
    # for i in range(0, num_train):
    #     print(f'printing testing case: {i}')
    #     file_name = dump_test+str(i)
    #     os.makedirs(file_name, exist_ok=True)
    #     for j in range(0, loading_steps):
    #         fig = plt.figure(constrained_layout=False, figsize=(11, 3))
    #         gs = fig.add_gridspec(1, 3)
    #         ax = fig.add_subplot(gs[0])
    #         h = ax.scatter(, cmap='coolwarm')
    #         ax.set_title('Testing case (Pred): ' + str(i))
    #         h.set_clim(vmin=0, vmax=10)
    #         fig.colorbar(h, ax=ax)

    #         ax = fig.add_subplot(gs[1])
    #         h = ax.imshow(y_true[count], cmap='coolwarm')
    #         ax.set_title('Testing case (True): ' + str(i))
    #         h.set_clim(vmin=0, vmax=10)
    #         fig.colorbar(h, ax=ax)

    #         ax = fig.add_subplot(gs[2])
    #         h = ax.imshow(abs(y_true[count] - y_pred[count]), cmap='coolwarm')
    #         ax.set_title('Testing case (Abs Diff): ' + str(i))
    #         h.set_clim(vmin=0, vmax=0.5)
    #         fig.colorbar(h, ax=ax)

    #         fig.savefig(file_name + '/' + str(j) + '.png')
    #         plt.close()

    #         count = count + 1

    phi_pred = model.forward(phi0_test_tensor, x_test_tensor)
    phi_pred = to_numpy(phi_pred)
    count = 0
    for i in range(0, num_test*loading_steps):

        x_plt = x_test[i*450:(i+1)*450, 0]
        y_plt = x_test[i*450:(i+1)*450, 1]
        phi0_plt = phi0_test[i*450, :]
        phi_true_plt = y_test[i*450:(i+1)*450, 0]
        phi_pred_plt = phi_pred[i*450:(i+1)*450, 0]

        fig = plt.figure(constrained_layout=False, figsize=(7, 7))
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[0])
        h = ax.scatter(x_plt, y_plt, c = phi_pred_plt, cmap='coolwarm')
        ax.set_title('Testing case (Pred): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = ax.scatter(x_plt, y_plt, c = phi_true_plt, cmap='coolwarm')
        ax.set_title('Testing case (True): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[2])
        h = ax.scatter(x_plt, y_plt, c = phi0_plt, cmap='coolwarm')
        ax.set_title('Testing case (init): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[3])
        h = ax.scatter(x_plt, y_plt, c = abs(phi_true_plt - phi_pred_plt), cmap='coolwarm')
        ax.set_title('Testing case (Abs Diff): ' + str(i))
        h.set_clim(vmin=0, vmax=0.1)
        fig.colorbar(h, ax=ax)

        fig.savefig(dump_test + '/img_' + str(i) + '.png')
        plt.close()

    ## save train
    phi_train = model.forward(phi0_train_tensor, x_train_tensor)
    phi_train = to_numpy(phi_train)
    count = 0
    for i in range(0, num_train*loading_steps):

        x_plt = x_train[i*450:(i+1)*450, 0]
        y_plt = x_train[i*450:(i+1)*450, 1]
        phi0_plt = phi0_train[i*450, :]
        phi_true_plt = y_train[i*450:(i+1)*450, 0]
        phi_pred_plt = phi_pred[i*450:(i+1)*450, 0]

        fig = plt.figure(constrained_layout=False, figsize=(7, 7))
        gs = fig.add_gridspec(2, 2)

        ax = fig.add_subplot(gs[0])
        h = ax.scatter(x_plt, y_plt, c = phi_pred_plt, cmap='coolwarm')
        ax.set_title('Testing case (Pred): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[1])
        h = ax.scatter(x_plt, y_plt, c = phi_true_plt, cmap='coolwarm')
        ax.set_title('Testing case (True): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[2])
        h = ax.scatter(x_plt, y_plt, c = phi0_plt, cmap='coolwarm')
        ax.set_title('Testing case (init): ' + str(i))
        h.set_clim(vmin=0, vmax=1)
        fig.colorbar(h, ax=ax)

        ax = fig.add_subplot(gs[3])
        h = ax.scatter(x_plt, y_plt, c = abs(phi_true_plt - phi_pred_plt), cmap='coolwarm')
        ax.set_title('Testing case (Abs Diff): ' + str(i))
        h.set_clim(vmin=0, vmax=0.1)
        fig.colorbar(h, ax=ax)

        fig.savefig(dump_train+ '/img_' + str(i) + '.png')
        plt.close()
    

    
if __name__ == "__main__":
    main()
