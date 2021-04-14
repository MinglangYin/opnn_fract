"""
Author: Minglang Yin, minglang_yin@brown.edu, Somdatta Goswami, somdatta_goswami@brown.edu
DeepONet for predicting damage field using data from Phase-field Models.
Model for obtaining the crack at the first displacement step.
"""
import os
import torch
import time
import numpy as np
import scipy.io as io
from functions import *
from mfnn import *
#from pinn import *
from preproc import *
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    device = args.device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_step = args.save_step
    batch_size = args.batch_size
    restart = args.restart

    dim_br = [1, 50, 50, 50]
    dim_tr = [3, 50, 50, 50]
    lmd = 0 #1e-5
    num_cols = 122
    num_rows = 122

    loading_steps = 8 # max = 8
    num_train = loading_steps
    
    Param = {}
    Param['num_cases'] = loading_steps
    
    ## create folders
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## dataset
    vDelta_train, x_train, u_train, v_train, phi_train = prep_train_test(Param)

    vDelta_train_tensor = torch.tensor(vDelta_train, dtype=torch.float).to(device)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float, requires_grad=True).to(device)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float, requires_grad=True).to(device)
    v_train_tensor = torch.tensor(v_train, dtype=torch.float, requires_grad=True).to(device)
    phi_train_tensor = torch.tensor(phi_train, dtype=torch.float, requires_grad=True).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    train_loss = np.zeros((args.epochs, 1))
    
    ## training
    def train(epoch, vDelta, x):
        model.train()
        def closure():
            optimizer.zero_grad()
#            y_out = model.forward(appDisp, x)
#            u_out, v_out, phi_out = y_out[:, 0], y_out[:, 1], y_out[:, 2]
#            loss_data = (1e5*(u_out - u[:, 0])**2).mean() + (1e4*(v_out - v[:, 0])**2).mean() + ((phi_out - phi[:, 0])**2).mean()
#            loss_data = ((u_out - u[:, 0])**2).mean() + ((v_out - v[:, 0])**2).mean() + ((phi_out - phi[:, 0])**2).mean()
            loss_pde = model.loss_energy(vDelta, x)

            print(f"epoch: {epoch}, loss_pde: {loss_pde}")
#            print(f"epoch: {epoch}, loss_data: {loss_data}, loss_pde: {loss_pde}")
            loss = loss_pde #+ 10*loss_data
            train_loss[epoch, 0] = loss
            loss.backward()
            return loss
        optimizer.step(closure)
 
    ## Iterations
    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):

        if epoch == epochs-10000:
            optimizer = torch.optim.LBFGS(model.parameters(), lr = 0.0001)

        ## Training
        train(epoch, vDelta_train_tensor, x_train_tensor)


    toc = time.time()

    print(f'total training time: {int((toc-tic)/60)} min', flush=True)
    np.savetxt('./train_loss.txt', train_loss)

#################################################################
    
    y_pred = model.forward(vDelta_train_tensor, x_train_tensor)
    u_pred, v_pred, phi_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    u_pred = to_numpy(u_pred).reshape(num_train, -1)
    v_pred = to_numpy(v_pred).reshape(num_train, -1)
    phi_pred = to_numpy(phi_pred).reshape(num_train, -1)
    u_true = u_train.reshape(num_train, -1)
    v_true = v_train.reshape(num_train, -1)
    phi_true = phi_train.reshape(num_train, -1)
    
    for i in range(0, num_train):
        
        damage_pred_print = phi_pred[i, :].reshape(num_rows, num_cols)
        damage_true_print = phi_true[i, :].reshape(num_rows, num_cols)
        xDisp_pred_print = u_pred[i, :].reshape(num_rows, num_cols)
        xDisp_true_print = u_true[i, :].reshape(num_rows, num_cols)
        yDisp_pred_print = v_pred[i, :].reshape(num_rows, num_cols)
        yDisp_true_print = v_true[i, :].reshape(num_rows, num_cols)

        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(10, 10))
        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.4)
        gs = fig.add_gridspec(3, 3)

        ax = fig.add_subplot(gs[0,0])
        h = ax.imshow(damage_pred_print,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case Phi (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,0])
        h = ax.imshow(damage_true_print,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)        
        ax.set_title('Training case Phi (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,0])
        h = ax.imshow(abs(damage_pred_print - damage_true_print),origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case Phi (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,1])
        h = ax.imshow(xDisp_pred_print,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case X_disp (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,1])
        h = ax.imshow(xDisp_true_print,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case X_disp (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,1])
        h = ax.imshow(abs(xDisp_pred_print - xDisp_true_print),origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case X_disp (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,2])
        h = ax.imshow(yDisp_pred_print,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case Y_disp (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,2])
        h = ax.imshow(yDisp_true_print,origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case Y_disp (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,2])
        h = ax.imshow(abs(yDisp_pred_print - yDisp_true_print),origin='lower')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('Training case Y_disp (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_train + '/img_' + str(i) + '.png')
        plt.close()
    
    ####################### Figure #######################
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    
    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss, color='blue', label='Training Loss')

    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig('./loss_his.png')
    
if __name__ == "__main__":
    main()
