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
    num_train = Param['num_train']
    num_test = Param['num_test']
    num_crackPos = Param['crack_pos']
    
    dataset = io.loadmat('dataset.mat')
    x = dataset['coord']
    hist_init = dataset['hist_init']
    phi = dataset['phi_train']
    u = dataset['u_train']
    v = dataset['v_train']
    stress_train = dataset['stress_train'].T  
    
    tmp_length = x.shape[0]
    num_gap = 1
    num_sample = int(tmp_length/num_gap)
    id_sample = np.linspace(0, num_sample-1, num_sample, dtype=int)

    x_new = np.zeros((num_sample*num_cases*num_crackPos, 6))
    appDisp_new = np.zeros((num_sample*num_cases*num_crackPos, 1))

    for i in range(0, num_cases*num_crackPos):
        x_new[i*num_sample:(i+1)*num_sample,0:2] = x[id_sample, 0:2]
        x_new[i*num_sample:(i+1)*num_sample,2:3] = u[id_sample, i:i+1]
        x_new[i*num_sample:(i+1)*num_sample,3:4] = v[id_sample, i:i+1]
        x_new[i*num_sample:(i+1)*num_sample,4:5] = phi[id_sample, i:i+1]
        x_new[i*num_sample:(i+1)*num_sample,5:6] = hist_init[id_sample, i:i+1]
        
        appDisp_new[i*num_sample:(i+1)*num_sample,:] = stress_train[:, i]

    ###########################################################################
    id_train = num_sample*num_train*num_crackPos
    id_test = num_sample*num_test*num_crackPos

    x_train = x_new[0:id_train, 0:2]
    u_train = x_new[0:id_train, 2:3]
    v_train = x_new[0:id_train, 3:4]
    phi_train = x_new[0:id_train, 4:5]
    hist_init_train = x_new[0:id_train, 5:6]
    
    appDisp_train = appDisp_new[0:id_train]

    x_test = x_new[id_train:id_train+id_test, 0:2]
    u_test = x_new[id_train:id_train+id_test, 2:3]
    v_test = x_new[id_train:id_train+id_test, 3:4]
    phi_test = x_new[id_train:id_train+id_test, 4:5]
    hist_init_test = x_new[id_train:id_train+id_test, 5:6]
    
    appDisp_test = appDisp_new[id_train:id_train+id_test, :]

    return appDisp_train, x_train, u_train, v_train, phi_train, hist_init_train, appDisp_test, x_test, u_test, v_test, phi_test, hist_init_test

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
