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
    loading_steps = Param['loading_steps']
    num_cases = Param['num_cases']
    num_train = Param['num_train']
    num_test = Param['num_test']
    size_filter = Param['size_filter']
    if_gauss_filter = Param['if_gauss_filter']
    
    num_snaps = loading_steps*num_cases

    ## load mat
    dataset = io.loadmat('../dataset.mat')
    phi_init_raw = dataset['phi_init_train']
    phi_raw = dataset['phi_train']
    img_size = 122

    phi_init = np.zeros((num_snaps, 1, img_size, img_size))
    phi = np.zeros((num_snaps, 1, img_size, img_size))

    for i in range(0, num_snaps):
        phi_init[i] = phi_init_raw[:, i].reshape(1, img_size, img_size)
        phi[i] = phi_raw[:, i].reshape(1, img_size, img_size)

    ######################################
    fig = plt.figure(constrained_layout=False, figsize=(7, 3))
    gs = fig.add_gridspec(1, 2)

    ax = fig.add_subplot(gs[0])
    h = ax.imshow(phi[0][0], cmap='coolwarm')
    ax.set_title('Testing case (Pred): ' + str(i))
    h.set_clim(vmin=0, vmax=1)
    fig.colorbar(h, ax=ax)

    ax = fig.add_subplot(gs[1])
    h = ax.imshow(phi[7][0], cmap='coolwarm')
    ax.set_title('Testing case (Pred): ' + str(i))
    h.set_clim(vmin=0, vmax=1)
    fig.colorbar(h, ax=ax)

    # fig.savefig('test.png')
    # plt.close()
    # exit()
    ######################################

    ######################
    x1 = dataset['x_train']
    x2 = dataset['y_train']
    force = dataset['stress_train']*100

    ## downsample
    if if_gauss_filter == 1:
        phi_init = block_reduce(phi_init, block_size=(1, 1, size_filter, size_filter), func=np.mean)
        phi = block_reduce(phi, block_size=(1, 1, size_filter, size_filter), func=np.mean)
        x1 = block_reduce(x1, block_size=(1, size_filter), func=np.mean)
        x2 = block_reduce(x2, block_size=(1, size_filter), func=np.mean)

    length_img = int(np.ceil(img_size/size_filter))
    len_img_sq = length_img*length_img

    #######################################################################
    xx1, xx2 = np.meshgrid(x1, x2)
    x1_flat = xx1.flatten()
    x2_flat = xx2.flatten()
    tmp_length = x1_flat.shape[0]

    ## x_new
    x_new = np.zeros((tmp_length*num_snaps, 3))
    for i in range(0, num_snaps):
        x_new[i*tmp_length:(i+1)*tmp_length, 0] = x1_flat
        x_new[i*tmp_length:(i+1)*tmp_length, 1] = x2_flat
    for i in range(0, num_cases):
        id_left = i*tmp_length*loading_steps
        for j in range(0, loading_steps):
            x_new[id_left + j*tmp_length : id_left + (j+1)*tmp_length, 2] = force[j]

    ## images
    phi_init_new = np.zeros((tmp_length*num_snaps, 1, length_img, length_img))
    # exit()

    for i in range(0, num_snaps):
        phi_init_new[i*tmp_length:(i+1)*tmp_length] = phi_init[i]
    phi_new = phi.flatten()[:, None]

    ###########################################################################
    id_train = num_train*len_img_sq*loading_steps
    id_test = num_test*len_img_sq*loading_steps

    x_train = x_new[0:id_train]
    y_train = phi_new[0:id_train]
    images_train = phi_init_new[:id_train]

    x_test = x_new[id_train:id_train+id_test]
    y_test = phi_new[id_train:id_train+id_test]
    images_test = phi_init_new[id_train:id_train+id_test]

    return images_train, x_train, y_train, images_test, x_test, y_test
