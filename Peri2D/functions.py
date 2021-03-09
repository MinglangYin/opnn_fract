"""
Author: Minglang Yin, minglang_yin@brown.edu
"""
import numpy as np
import torch
import argparse
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def ParseArgument():
    parser = argparse.ArgumentParser(description='DeepONet')
    parser.add_argument('--batch-size', type=int, default=500000, metavar='N',
                        help = 'input batch size for low-fidelity training')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help = 'number of epochs to train (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda', metavar='N',
                        help = 'computing device (default: GPU)')
    parser.add_argument('--save-step', type=int, default=100, metavar='N',
                        help = 'save_step (default: 200)')
    parser.add_argument('--restart', type=int, default=0, metavar='N',
                        help = 'if restart (default: 0)')

    args = parser.parse_args()
    return args

def CurveFit(x_data, y_data, x_new):
    spl = UnivariateSpline(x_data, y_data)
    spl.set_smoothing_factor(0.5)
    y_new = spl(x_new)
    return y_new

def Extend(image, x, y):
    n_s, n_pt = x.shape
    images = np.zeros((n_s*n_pt, 1, image.shape[2], image.shape[3]))
    
    xx = x.reshape(-1)
    yy = y.reshape(-1)
    for i in range(0, n_s):
        images[i*n_pt:(i+1)*n_pt] = image[i]

    return images, xx, yy

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def plot_loss():
    train_loss = np.loadtxt('./train_loss.txt')
    test_loss = np.loadtxt('./test_loss.txt')
    # wod_tear_res = np.loadtxt('./wod_tear_err.txt')
    
    # train_loss = np.mean(train_loss, axis = 1)
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    
    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss, color='blue', label='Training Loss')
    ax.plot(x, test_loss, color='red', label='Test Loss', linestyle='dashed')

    # ax.plot(x, wod_tear_res[:, 0], color='black', label='WOD Loss', linestyle='dashed')
    # ax.plot(x, wod_tear_res[:, 1], color='orange', label='P_tear Loss', linestyle='dashed')
    # ax.axhline(0.05, linestyle='dashed')
    # ax.set_xlim([0, 2])
    # ax.set_ylim([0, 100])
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig('./loss_his.png')

def Regreesion(model_name, x_train, y_train, x_test):
    if model_name == 'Lasso' or model_name == 'lasso':
        model = Lasso(fit_intercept=True)
    elif model_name == 'Ridge' or model_name == 'ridge':
        model = Ridge(fit_intercept=True)
    elif model_name == 'Linear' or model_name == 'linear':
        model = LinearRegression(fit_intercept=True)
    else:
        print('Model not included!')
        exit(-1)

    y_pred = np.zeros((x_test.shape[0], y_train.shape[1]))
    for i in range(0, y_train.shape[1]):
        model.fit(x_train, y_train[:, i])
        y_pred[:, i] = model.predict(x_test)

    y_pred_train = np.zeros((x_train.shape[0], y_train.shape[1]))
    for i in range(0, y_train.shape[1]):
        model.fit(x_train, y_train[:, i])
        y_pred_train[:, i] = model.predict(x_train)

    return y_pred, y_pred_train

if __name__ == "__main__":
    plot_loss()