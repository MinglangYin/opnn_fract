"""
Author: Minglang Yin, minglang_yin@brown.edu
DeepONet for predicting pressure-volume curve.
"""
import os
import torch
import time
import math
import numpy as np


from functions import *
from preprocess import *
from opnn import *

import matplotlib
matplotlib.use('Agg')

def main():
    ## parsing arguments
    args = ParseArgument()
    epochs = args.epochs
    # branch_dim = [1, 64, 128, 256]
    # trunk_dim = [1, 40, 40, 256]
    branch_dim = [1, 32, 64, 128]
    trunk_dim = [1, 40, 40, 40, 128]
    
    file_start = 201
    file_end = 600
    p_train = 0.95
    lmd = 0.0005
    file_test = math.floor((file_end + 1 - file_start)*(1 - p_train))
    file_train = math.floor((file_end + 1 - file_start)*p_train)

    dump_test = './Predictions/x20/Test/'
    dump_train = './Predictions/x20/Train/'
    model_path = 'CheckPts/model_chkpts.pt'
    check_step = 500
    part_img = 1
    
    ## pre-processing
    data_path = '../../../Data/Clean/x20/'
    dataset = PreProc(data_path, file_start, file_end) 

    images = dataset['images']
    x = dataset['volume']
    y = dataset['pres']

    ## if use partial images
    if part_img == 1:
        images = images[:, :, :20, :10]
        images[:, :, :16, :5] = 0


    x_length = x.shape[0]
    num_train = int(p_train*x_length)
    num_test = int((1 - p_train)*x_length)

    n_l, n_r = 110, 130
    x_plot = x[0:200]
    x_test = x[n_l*200:130*200]
    y_test = y[n_l*200:130*200]
    img_test = images[n_l*200:130*200]

    x_train = np.concatenate((x[:n_l*200, :], x[130*200:, :]), axis=0)
    y_train = np.concatenate((y[:n_l*200, :], y[130*200:, :]), axis=0)
    img_train = np.concatenate((images[:n_l*200], images[130*200:]), axis=0)

    ## tensorizing
    device = torch.device(args.device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    img_test = torch.tensor(img_test, dtype=torch.float).to(device)
    
    # ## initialization
    model = OpNN(branch_dim, trunk_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## plotting
    WOD = np.zeros((20, 3))

    ########### test############
    y_pred = model(img_test, x_test)
    y_pred = to_numpy(y_pred).reshape(file_test, -1)
    y_true = y_test.reshape(file_test, -1)

    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    for i in range(0, 20):
        ax = plt.subplot(5, 4, i+1)
        plt.tight_layout()
        ax.plot(x_plot, y_pred[i, :], color='blue', label='Model')
        ax.plot(x_plot, y_true[i, :], color='red', label='Data', linestyle='dashed')

        ##WOD
        WOD[i, 0] = y_true[i, :].mean()*(x_plot[-1] - x_plot[0])
        WOD[i, 1] = y_pred[i, :].mean()*(x_plot[-1] - x_plot[0])

        ax.set_xlim([0, 2])
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pressure')
        ax.set_xlabel('Volume')
        ax.set_title('Test case: ' + str(i))
        ax.legend(loc='upper left')
    fig.savefig('./Predictions/test.png')
    plt.close()

    ########### Plot WOD/P_tear/P_max##########
    x = np.linspace(60, 120)
    fig = plt.figure(constrained_layout=False, figsize=(8, 12))
    plt.tight_layout()
    ##WOD
    ax = plt.subplot(3, 2, 1)
    ax.scatter(WOD[:, 0], WOD[:, 1], color='blue')
    ax.plot(x, x, color='black', alpha = 0.3)
    ax.set_xlim([60, 120])
    ax.set_ylim([60, 120])
    ax.set_ylabel('$WOD_{pred}(J)$')
    ax.set_xlabel('$WOD_{data}(J)$')
    ax.set_title('(a) Work of Dissection')
    ax.set_aspect(1)

    ax = plt.subplot(3, 2, 2)
    diff = WOD[:, 1]-WOD[:, 0]
    diff_mean = diff.mean()
    ax.scatter(WOD[:, 0], diff, color='blue', linestyle='--')
    ax.axhline(diff_mean, linestyle='--')
    ax.fill_between(x, diff_mean+np.sqrt(diff.var()), diff_mean-np.sqrt(diff.var()), color = 'gray', alpha = 0.2, label = 'std')
    ax.set_xlim([60, 120])
    ax.set_ylim([-30, 30])
    ax.set_ylabel('Error(J)')
    ax.set_xlabel('$WOD_{data}(J)$')
    ax.legend()
    ax.set_title('(b)')
    ax.set_aspect(1)

    ##P_tear
    x = [30, 80]
    diff = y_pred[:, 109] - y_true[:, 109]
    diff_mean = diff.mean()
    ax = plt.subplot(3, 2, 3)
    ax.scatter(y_true[:, 109], y_pred[:, 109], color='blue')
    ax.plot(x, x, color='black', alpha = 0.3)
    ax.set_xlim(x)
    ax.set_ylim(x)
    ax.set_ylabel('$P_{pred}$')
    ax.set_xlabel('$P_{data}$')
    ax.set_title('(c) $P_{tear}$(kPa)')
    ax.set_aspect(1)

    ax = plt.subplot(3, 2, 4)
    ax.scatter(y_true[:, 109], diff, color='blue')
    ax.fill_between(x, diff_mean+np.sqrt(diff.var()), diff_mean-np.sqrt(diff.var()), color = 'gray', alpha = 0.2, label = 'std')
    ax.plot(x, x, color='black', alpha = 0.3)
    plt.axhline(diff_mean, linestyle='--')
    ax.set_xlim(x)
    ax.set_ylim([-25, 25])
    ax.set_ylabel('$Error$')
    ax.set_xlabel('$P_{data}$')
    ax.legend()
    ax.set_title('(d)')
    ax.set_aspect(1)

    ##P_max
    x = [40, 80]
    diff = y_pred.max(axis=1) - y_true.max(axis=1)
    diff_mean = diff.mean()
    ax = plt.subplot(3, 2, 5)
    ax.scatter(y_true.max(axis=1), y_pred.max(axis=1), color='blue')
    ax.plot(x, x, color='black', alpha = 0.3)
    ax.set_xlim(x)
    ax.set_ylim(x)
    ax.set_ylabel('$P_{pred}$')
    ax.set_xlabel('$P_{data}$')
    ax.set_title('(e) $P_{max}$(kPa)')
    ax.set_aspect(1)

    ax = plt.subplot(3, 2, 6)
    ax.scatter(y_true.max(axis=1), diff, color='blue')
    ax.fill_between(x, diff_mean+np.sqrt(diff.var()), diff_mean-np.sqrt(diff.var()), color = 'gray', alpha = 0.2, label = 'std')
    ax.plot(x, x, color='black', alpha = 0.3)
    plt.axhline(diff_mean, linestyle='--')
    ax.set_xlim(x)
    ax.set_ylim([-20, 20])
    ax.set_ylabel('$Error$')
    ax.set_xlabel('$P_{data}$')
    ax.legend()
    ax.set_title('(f)')
    ax.set_aspect(1)


    plt.tight_layout()
    fig.savefig('./Predictions/Altman.png')
    plt.close()


    ########## train#########
    x_test = torch.tensor(x_train, dtype=torch.float).to(device)
    img_test = torch.tensor(img_train, dtype=torch.float).to(device)
    y_pred = model(img_test, x_test)
    y_pred = to_numpy(y_pred).reshape(file_train, -1)
    y_true = y_train.reshape(file_train, -1)

    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    for i in range(0, 20):
        ax = plt.subplot(5, 4, i+1)
        plt.tight_layout()
        ax.plot(x_plot, y_pred[i, :], color='blue', label='Model')
        ax.plot(x_plot, y_true[i, :], color='red', label='Data', linestyle='dashed')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pressure')
        ax.set_xlabel('Volume')
        ax.set_title('Train case: ' + str(i))
        ax.legend(loc='upper left')
    fig.savefig('./Predictions/train.png')
    plt.close()

    for i in range(0, file_train):
        # print(f'printing training case {i}')

        fig = plt.figure(constrained_layout=False, figsize=(6, 6))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x_plot, y_pred[i, :], color='blue', label='Model')
        ax.plot(x_plot, y_true[i, :], color='red', label='Data', linestyle='dashed')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 100])
        ax.set_ylabel('Pressure')
        ax.set_xlabel('Volume')
        ax.set_title('Train case: ' + str(201+i))
        ax.legend(loc = 'upper left')
        fig.savefig(dump_train + str(i+201)+'.png')
        plt.close()




if __name__ == "__main__":
    main()
