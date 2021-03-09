import torch
import scipy.io as io
import math
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
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

def prep_train_test(Param):
    #loading_steps = Param['loading_steps']
    num_cases = Param['num_cases']
    num_train = Param['num_train']
    num_test = Param['num_test']
    #size_filter = Param['size_filter']
    #if_gauss_filter = Param['if_gauss_filter']
    
    #num_snaps = loading_steps*num_cases

    ## load mat
    dataset = io.loadmat('dataset.mat')
    #phi_init_raw = dataset['phi_init_train']
    #phi_raw = dataset['phi_train']

    #phi_init = np.zeros((num_snaps, 1, 110, 110))
    #phi = np.zeros((num_snaps, 1, 110, 110))

    #for i in range(0, num_snaps):
        #phi_init[i] = phi_init_raw[:, i].reshape(1, 110, 110)
        #phi[i] = phi_raw[:, i].reshape(1, 110, 110)

    ######################################
    #fig = plt.figure(constrained_layout=False, figsize=(7, 3))
    #gs = fig.add_gridspec(1, 2)

#    ax = fig.add_subplot(gs[0])
#    h = ax.imshow(phi[0][0], cmap='coolwarm')
#    ax.set_title('Testing case (Pred): ' + str(i))
#    h.set_clim(vmin=0, vmax=1)
#    fig.colorbar(h, ax=ax)
#
#    ax = fig.add_subplot(gs[1])
#    h = ax.imshow(phi[7][0], cmap='coolwarm')
#    ax.set_title('Testing case (Pred): ' + str(i))
#    h.set_clim(vmin=0, vmax=1)
#    fig.colorbar(h, ax=ax)
#
#    fig.savefig('test.png')
#    plt.close()
#    exit()
    ######################################

    ######################
    x1 = dataset['x_train']
    x2 = dataset['y_train']
    y = dataset['damage']
    u = dataset['boundary_train']

    # for i in range(0, 100):
    #     print(i)
    #     fig = plt.figure(constrained_layout=False, figsize=(12, 6))
    #     gs = fig.add_gridspec(1, 1)
    #     ax = fig.add_subplot(gs[0])
    #     h = ax.scatter(x1, x2, c = y[:, i:i+1], cmap='coolwarm')
    #     ax.set_title('Testing case (Pred): ' + str(i))
    #     h.set_clim(vmin=0, vmax=1)
    #     fig.colorbar(h, ax=ax)
    #     fig.savefig('Predictions/Train/img_' + str(i) + '.png')
    #     plt.close()
    # exit()



    ## downsample
#    if if_gauss_filter == 1:
#        phi_init = block_reduce(phi_init, block_size=(1, 1, size_filter, size_filter), func=np.mean)
#        phi = block_reduce(phi, block_size=(1, 1, size_filter, size_filter), func=np.mean)
#        x1 = block_reduce(x1, block_size=(1, size_filter), func=np.mean)
#        x2 = block_reduce(x2, block_size=(1, size_filter), func=np.mean)
#
#    length_img = int(np.ceil(110/size_filter))
#    len_img_sq = length_img*length_img

    #######################################################################
#    xx1, xx2 = np.meshgrid(x1, x2)
#    x1_flat = xx1.flatten()
#    x2_flat = xx2.flatten()
    
#    for i in range(0, num_snaps):
#        x_new[i*tmp_length:(i+1)*tmp_length, 0] = x1_flat
#        x_new[i*tmp_length:(i+1)*tmp_length, 1] = x2_flat

    tmp_length = x1.shape[0]
    num_gap = 1
    num_sample = int(tmp_length/num_gap)
    # id_sample = np.random.choice(tmp_length, num_sample, replace=False)
    id_sample = np.linspace(0, num_sample-1, num_sample, dtype=int)

    x_new = np.zeros((num_sample*num_cases, 3))
    u_new = np.zeros((num_sample*num_cases, 10))

    for i in range(0, num_cases):
        x_new[i*num_sample:(i+1)*num_sample,0:1] = x1[id_sample, 0:1]
        x_new[i*num_sample:(i+1)*num_sample,1:2] = x2[id_sample, 0:1]
        x_new[i*num_sample:(i+1)*num_sample,2:3] = y[id_sample, i:i+1]

        u_new[i*num_sample:(i+1)*num_sample,:] = u[::5, i]

#        id_left = i*tmp_length*loading_steps
#        for j in range(0, loading_steps):
#            x_new[id_left + j*tmp_length : id_left + (j+1)*tmp_length, 2] = force[j]

    ## images
#    phi_init_new = np.zeros((tmp_length*num_snaps, 1, length_img, length_img))
    # exit()

#    for i in range(0, num_snaps):
#        phi_init_new[i*tmp_length:(i+1)*tmp_length] = phi_init[i]
#    phi_new = phi.flatten()[:, None]

    ###########################################################################
    id_train = num_sample*num_train
    id_test = num_sample*num_test

    x_train = x_new[0:id_train, 0:2]
    y_train = x_new[0:id_train, 2:3]
    u_train = u_new[0:id_train]

    x_test = x_new[id_train:id_train+id_test, 0:2]
    y_test = x_new[id_train:id_train+id_test, 2:3]
    u_test = u_new[id_train:id_train+id_test, :]

    return u_train, x_train, y_train, u_test, x_test, y_test
