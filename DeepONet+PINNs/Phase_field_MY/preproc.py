import torch
import scipy.io as io
import math
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
#from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

class pvstrut_dataset(Dataset):
    def __init__(self, images, x, y, device):
        self.img = torch.tensor(images, dtype=torch.float).to(device)
        self.x = torch.tensor(x, dtype=torch.float).to(device)
        self.y = torch.tensor(y, dtype=torch.float).to(device)
        self.len = images.shape[0]

    def __getitem__(self, index):
        return self.img[index], self.x[index], self.y[index]
        
    def __len__(self):
        return self.len

## damage
def prep_train_test(Param):   
    

    num_cases = Param['num_cases']
#    num_train = Param['num_train']
#    num_test = Param['num_test']
#    num_crackPos = Param['crack_pos']    
    
    constB = 92
    dataset_all = io.loadmat('dataset.mat')
    x = dataset_all['coord']
    phi = dataset_all['phi_train']
    u = dataset_all['u_train']
    v = dataset_all['v_train']
    stress_train = dataset_all['stress_train'].T

    dataset_hist = io.loadmat('history.mat')
    hist = dataset_hist['hist']
    hist[:,0:1] = constB*hist[:,0:1]
#    i = 35
#    fig = plt.figure(figsize=(5, 5))
#    plt.scatter(x[:,0], x[:,1], c=hist_init[:,i])
#    plt.show()
#    
##    hist1 = hist_init[:,i:i+1]  
##    fig = plt.figure(figsize=(5, 5))
##    hist_print1d = hist_init[:,i].reshape(122,-1,order = 'F').transpose()
##    hist_print1d = hist_print1d.ravel()
###    hist_print1d = hist_print1d.reshape(122*122,-1)
##    plt.scatter(x[:,0], x[:,1], c=hist_print1d)
#    
#    fig = plt.figure(figsize=(5, 5))
#    hist_check = hist_init[:,i].reshape(122,-1)
#    plt.imshow(hist_check,origin='lower')
#    plt.show()
#    hist_print = hist_init[:,i].reshape(122,122)
#    plt.imshow(hist_print,origin='lower',interpolation='bilinear')
#    plt.show()
    
    tmp_length = x.shape[0]
    num_gap = 1
    num_sample = int(tmp_length/num_gap)
    id_sample = np.linspace(0, num_sample-1, num_sample, dtype=int)

    x_new = np.zeros((num_sample*num_cases, 6))
    appDisp_new = np.zeros((num_sample*num_cases, 1))

    for i in range(0, num_cases):
        x_new[i*num_sample:(i+1)*num_sample,0:2] = x[id_sample, 0:2]
        x_new[i*num_sample:(i+1)*num_sample,2:3] = hist[id_sample, i:i+1]
        x_new[i*num_sample:(i+1)*num_sample,3:4] = u[id_sample, i:i+1]
        x_new[i*num_sample:(i+1)*num_sample,4:5] = v[id_sample, i:i+1]
        x_new[i*num_sample:(i+1)*num_sample,5:6] = phi[id_sample, i:i+1]
        
        
        appDisp_new[i*num_sample:(i+1)*num_sample,:] = stress_train[:, i]

    ###########################################################################
    
#    id_train = num_sample*num_train*num_crackPos
#    id_test = num_sample*num_test*num_crackPos

    x_train = x_new[:, 0:3]
    u_train = x_new[:, 3:4]
    v_train = x_new[:, 4:5]
    phi_train = x_new[:, 5:6]
    
    appDisp_train = appDisp_new[:,0:1]


    return appDisp_train, x_train, u_train, v_train, phi_train

## uv error
# def prep_train_test(Param):
#     num_cases = Param['num_cases']
#     num_train = Param['num_train']
#     num_test = Param['num_test']

#     ## load mat
#     dataset = io.loadmat('dataset.mat')
#     x1 = dataset['x_train']
#     x2 = dataset['y_train']
#     u = dataset['boundary_train']

#     dataset = io.loadmat('dataset_uv.mat')
#     y = dataset['uError']

#     ## Gaussian Filter
#     # y_new = np.zeros((y.shape[0], y.shape[1]))
#     # for i in range(0, num_cases):
#     #     y_tmp = gaussian_filter( y[:, i].reshape(193, 74), sigma = 5)
#     #     y_new[:, i] = y_tmp.flatten()

#     # y = y_new

#     tmp_length = x1.shape[0]
#     num_gap = 1
#     num_sample = int(tmp_length/num_gap)
#     # id_sample = np.random.choice(tmp_length, num_sample, replace=False)
#     id_sample = np.linspace(0, num_sample-1, num_sample, dtype=int)

#     x_new = np.zeros((num_sample*num_cases, 3))
#     u_new = np.zeros((num_sample*num_cases, 25))

#     for i in range(0, num_cases):
#         x_new[i*num_sample:(i+1)*num_sample,0:1] = x1[id_sample, 0:1]
#         x_new[i*num_sample:(i+1)*num_sample,1:2] = x2[id_sample, 0:1]
#         x_new[i*num_sample:(i+1)*num_sample,2:3] = y[id_sample, i:i+1]

#         u_new[i*num_sample:(i+1)*num_sample,:] = u[::2, i]

#     ###########################################################################
#     id_train = num_sample*num_train
#     id_test = num_sample*num_test

#     x_train = x_new[0:id_train, 0:2]
#     y_train = x_new[0:id_train, 2:3]
#     u_train = u_new[0:id_train]

#     x_test = x_new[id_train:id_train+id_test, 0:2]
#     y_test = x_new[id_train:id_train+id_test, 2:3]
#     u_test = u_new[id_train:id_train+id_test, :]

#     return u_train, x_train, y_train, u_test, x_test, y_test
