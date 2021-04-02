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
    device = "cpu"#args.device
    save_step = args.save_step
    batch_size = args.batch_size
    restart = args.restart

    dim_br = [1, 50, 50, 50]
    dim_tr = [2, 50, 50, 50]
    lmd = 0 #1e-5

    loading_steps = 3 # max = 8
    num_train = 2
    num_test = 1
    num_crackPos = 9 # DO not change this 
    
    Param = {}
    Param['num_cases'] = loading_steps
    Param['num_train'] = num_train
    Param['num_test'] = num_test
    Param['crack_pos'] = num_crackPos
    
    ## create folders
    dump_test = './Predictions/Test/'
    dump_train = './Predictions/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset
    appDisp_train, x_train, u_train, v_train, phi_train, hist_init_train, appDisp_test, x_test, u_test, v_test, phi_test, hist_init_test = prep_train_test(Param)
    # print(f"appDisp_train: {appDisp_train.shape}")
    # print(f"x_train: {x_train.shape}")
    # print(f"u_train: {u_train.shape}")
    # print(f"v_train: {v_train.shape}")
    # print(f"phi_train: {phi_train.shape}")
    # print(f"hist_init_train: {hist_init_train.shape}")
    # print(f"appDisp_test: {appDisp_test.shape}")
    # print(f"x_test: {x_test.shape}")
    # print(f"u_test: {u_test.shape}")
    # print(f"v_test: {v_test.shape}")
    # print(f"phi_test: {phi_test.shape}")
    # print(f"hist_init_test: {hist_init_test.shape}")
    # scipy.io.savemat('Test.mat',{'AppDisp':appDisp_test, 'x':x_test, 'u':u_test, 'v':v_test, 'phi':phi_test, 'hist_init':hist_init_test})
    # scipy.io.savemat('Train.mat',{'AppDisp':appDisp_train, 'x':x_train, 'u':u_train, 'v':v_train, 'phi':phi_train, 'hist_init':hist_init_train})
    # exit()
    ##
    appDisp_train_tensor = torch.tensor(appDisp_train, dtype=torch.float).to(device)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float, requires_grad=True).to(device)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float, requires_grad=True).to(device)
    v_train_tensor = torch.tensor(v_train, dtype=torch.float, requires_grad=True).to(device)
    phi_train_tensor = torch.tensor(phi_train, dtype=torch.float, requires_grad=True).to(device)
    hist_init_train_tensor = torch.tensor(hist_init_train, dtype=torch.float).to(device)
    
    appDisp_test_tensor = torch.tensor(appDisp_test, dtype=torch.float).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float, requires_grad=True).to(device)
    u_test_tensor = torch.tensor(u_test, dtype=torch.float, requires_grad=True).to(device)
    v_test_tensor = torch.tensor(v_test, dtype=torch.float, requires_grad=True).to(device)
    phi_test_tensor = torch.tensor(phi_test, dtype=torch.float, requires_grad=True).to(device)
    hist_init_test_tensor = torch.tensor(hist_init_test, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br, dim_tr).to(device)
    model = model.float()    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    train_loss = np.zeros((args.epochs, 1))
    test_loss = np.zeros((args.epochs, 1))
    
    ## training
    def train(epoch, appDisp, x, u, v, phi, hist_init):
        model.train()

        def closure():
            optimizer.zero_grad()
            
            u_out = model.forward_u(appDisp, x)
            # exit()
            v_out = model.forward_v(appDisp, x)
            phi_out = model.forward_phi(appDisp, x)
            # exit()
            loss_data = ((u_out - u[:,0])**2).mean() + ((v_out - v[:,0])**2).mean() + ((phi_out - phi[:,0])**2).mean()
            #print(f"loss_data: {loss_data}")
            # exit()
            loss_pde = model.loss_energy(appDisp, x, hist_init)
            # print(f"loss_pde: {loss_pde}")
            # exit()
            loss = loss_pde + loss_data
            # print(f"loss: {loss}")
            # exit()
            train_loss[epoch, 0] = loss
            loss.backward()
            return loss
        optimizer.step(closure)
 
    ## Iterations
    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):
        
        ## Training
        train(epoch, appDisp_train_tensor, x_train_tensor, u_train_tensor, v_train_tensor, phi_train_tensor, hist_init_train_tensor)
        # exit(0)
        ## Testing
        u_pred = to_numpy(model.forward_u(appDisp_test_tensor, x_test_tensor))
        v_pred = to_numpy(model.forward_v(appDisp_test_tensor, x_test_tensor))
        phi_pred = to_numpy(model.forward_phi(appDisp_test_tensor, x_test_tensor))
        u_true = to_numpy(u_test_tensor)
        v_true = to_numpy(v_test_tensor)
        phi_true = to_numpy(phi_test_tensor)
        test_loss[epoch, 0] = ((u_pred - u_true[:,0])**2).mean() + ((v_pred - v_true[:,0])**2).mean() + ((phi_pred - phi_true[:,0])**2).mean()
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}', flush=True)
        # exit(0)
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
    u_pred = model.forward_u(appDisp_test_tensor, x_test_tensor)
    v_pred = model.forward_v(appDisp_test_tensor, x_test_tensor)
    phi_pred = model.forward_phi(appDisp_test_tensor, x_test_tensor)
    # print(f"u_pred: {u_pred.shape}")
    # print(f"phi_pred: {v_pred.shape}")
    # print(f"phi_pred: {phi_pred.shape}")
    # scipy.io.savemat('Pred.mat',{'u':u_pred, 'v':v_pred, 'phi':phi_pred})
    # exit(0)    
    u_pred = to_numpy(u_pred).reshape(num_test*num_crackPos, -1)
    v_pred = to_numpy(v_pred).reshape(num_test*num_crackPos, -1)
    phi_pred = to_numpy(phi_pred).reshape(num_test*num_crackPos, -1)
    u_true = u_test.reshape(num_test*num_crackPos, -1)
    v_true = v_test.reshape(num_test*num_crackPos, -1)
    phi_true = phi_test.reshape(num_test*num_crackPos, -1)
    # print(f"u_pred_reshape: {u_pred.shape}")
    # print(f"phi_pred_reshape: {v_pred.shape}")
    # print(f"phi_pred_reshape: {phi_pred.shape}")
    # print(f"u_pred_reshape: {u_true.shape}")
    # print(f"phi_pred_reshape: {v_true.shape}")
    # print(f"phi_pred_reshape: {phi_true.shape}")
    # scipy.io.savemat('Pred_true.mat',{'u':u_pred, 'v':v_pred, 'phi':phi_pred, 'uT':u_true, 'vT':v_true, 'phiT':phi_true})  
    # exit(0)    
    num_cols = 122
    num_rows = 122
    count = 0
    for i in range(0, num_test*num_crackPos):
        damage_pred_print = phi_pred[i, :].reshape(num_rows, num_cols)
        damage_true_print = phi_true[i, :].reshape(num_rows, num_cols)
        xDisp_pred_print = u_pred[i, :].reshape(num_rows, num_cols)
        xDisp_true_print = u_true[i, :].reshape(num_rows, num_cols)
        yDisp_pred_print = v_pred[i, :].reshape(num_rows, num_cols)
        yDisp_true_print = v_true[i, :].reshape(num_rows, num_cols)

        print(f"printing case: {i}")
        fig = plt.figure(constrained_layout=False, figsize=(10, 10))
        gs = fig.add_gridspec(3, 3)
        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.4)
        
        ax = fig.add_subplot(gs[0,0])        
        h = ax.imshow(damage_pred_print)
        ax.set_title('Testing case Phi (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,0])
        h = ax.imshow(damage_true_print)
        ax.set_title('Testing case Phi (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,0])
        h = ax.imshow(abs(damage_pred_print - damage_true_print))
        ax.set_title('Testing case Phi (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,1])
        h = ax.imshow(xDisp_pred_print)
        ax.set_title('Testing case X_disp (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,1])
        h = ax.imshow(xDisp_true_print)
        ax.set_title('Testing case X_disp (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,1])
        h = ax.imshow(abs(xDisp_pred_print - xDisp_true_print))
        ax.set_title('Testing case X_disp (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,2])
        h = ax.imshow(yDisp_pred_print)
        ax.set_title('Testing case Y_disp (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,2])
        h = ax.imshow(yDisp_true_print)
        ax.set_title('Testing case Y_disp (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,2])
        h = ax.imshow(abs(yDisp_pred_print - yDisp_true_print))
        ax.set_title('Testing case Y_disp (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_test + '/img_' + str(i) + '.png')
        plt.close()

    # exit(0)
    ## save train
    
    u_pred = model.forward_u(appDisp_train_tensor, x_train_tensor)
    v_pred = model.forward_v(appDisp_train_tensor, x_train_tensor)
    phi_pred = model.forward_phi(appDisp_train_tensor, x_train_tensor)
    # print(f"u_pred: {u_pred.shape}")
    # print(f"phi_pred: {v_pred.shape}")
    # print(f"phi_pred: {phi_pred.shape}")
    # scipy.io.savemat('Pred.mat',{'u':u_pred, 'v':v_pred, 'phi':phi_pred})
    # exit(0) 
    
    u_pred = to_numpy(u_pred).reshape(num_train*num_crackPos, -1)
    v_pred = to_numpy(v_pred).reshape(num_train*num_crackPos, -1)
    phi_pred = to_numpy(phi_pred).reshape(num_train*num_crackPos, -1)
    u_true = u_train.reshape(num_train*num_crackPos, -1)
    v_true = v_train.reshape(num_train*num_crackPos, -1)
    phi_true = phi_train.reshape(num_train*num_crackPos, -1)
    # print(f"u_pred_reshape: {u_pred.shape}")
    # print(f"phi_pred_reshape: {v_pred.shape}")
    # print(f"phi_pred_reshape: {phi_pred.shape}")
    # print(f"u_pred_reshape: {u_true.shape}")
    # print(f"phi_pred_reshape: {v_true.shape}")
    # print(f"phi_pred_reshape: {phi_true.shape}")
    # scipy.io.savemat('Pred_true.mat',{'u':u_pred, 'v':v_pred, 'phi':phi_pred, 'uT':u_true, 'vT':v_true, 'phiT':phi_true})  
    # exit(0) 

    count = 0
    for i in range(0, num_train*num_crackPos):
        
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
        h = ax.imshow(damage_pred_print)
        ax.set_title('Training case Phi (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,0])
        h = ax.imshow(damage_true_print)
        ax.set_title('Teraining case Phi (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,0])
        h = ax.imshow(abs(damage_pred_print - damage_true_print))
        ax.set_title('Training case Phi (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,1])
        h = ax.imshow(xDisp_pred_print)
        ax.set_title('Training case X_disp (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,1])
        h = ax.imshow(xDisp_true_print)
        ax.set_title('Training case X_disp (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,1])
        h = ax.imshow(abs(xDisp_pred_print - xDisp_true_print))
        ax.set_title('Training case X_disp (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[0,2])
        h = ax.imshow(yDisp_pred_print)
        ax.set_title('Training case Y_disp (Pred): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        ax = fig.add_subplot(gs[1,2])
        h = ax.imshow(yDisp_true_print)
        ax.set_title('Training case Y_disp (True): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)
        
        ax = fig.add_subplot(gs[2,2])
        h = ax.imshow(abs(yDisp_pred_print - yDisp_true_print))
        ax.set_title('Training case Y_disp (Err): ' + str(i))
        # h.set_clim(vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax)

        fig.savefig(dump_train + '/img_' + str(i) + '.png')
        plt.close()
    
if __name__ == "__main__":
    main()
