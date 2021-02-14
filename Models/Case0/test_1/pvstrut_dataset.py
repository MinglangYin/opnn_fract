import torch
import scipy.io as io
import math
import numpy as np

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

def prep_train_test(ParamDict):
    path = ParamDict['path']
    if_gauss_filter = ParamDict['if_gauss_filter']
    if_ds_img = ParamDict['if_ds_img']
    id_test = ParamDict['id_test']
    id_train = ParamDict['id_train']
    size_filter = ParamDict['size_filter']
    dev = ParamDict['dev']

    dataset = io.loadmat(path)
    x = dataset['data_vnext']
    run_steps = dataset['run_capa']
    y = dataset['data_prev_disp']
    y = y.reshape(y.shape[0], 1, 24, 24)
    strut = dataset['data_init']
    damage_int = dataset['data_prev']
    images = np.concatenate((strut.reshape(strut.shape[0], 1, 24, 24), damage_int.reshape(strut.shape[0], 1, 24, 24)), axis = 1)

    if if_gauss_filter == 1:
        images = gaussian_filter(images, sigma=1)
        for i, img in enumerate(images):
            images[i] = (img - img.min())/(img.max() - img.min())
    if if_ds_img == 1:
        images = block_reduce(images, block_size=(1, 1, size_filter, size_filter), func=np.mean)
        y = block_reduce(y, block_size=(1, 1, size_filter, size_filter), func=np.mean)
        length_img = int(24/size_filter)
        len_img_sq = length_img*length_img


    ## split
    num_train = run_steps[0:id_train].sum()
    x_train = x[:num_train]
    y_train = y[:num_train]
    images_train = images[:num_train]

    num_test = run_steps[id_train:id_train+id_test].sum()
    x_test = x[num_train:num_train+num_test]
    y_test = y[num_train:num_train+num_test]
    images_test = images[num_train:num_train+num_test]

    #######################################################################
    x = np.linspace(0, length_img, length_img)/length_img
    y = np.linspace(0, length_img, length_img)/length_img
    xx, yy = np.meshgrid(x, y)
    coord = np.concatenate((xx.reshape(-1)[:, None], yy.reshape(-1)[:, None]), axis=1)

    y_train_new = y_train.flatten()[:, None]
    x_train_new = np.zeros((y_train_new.shape[0], 3))
    images_train_new = np.zeros((y_train_new.shape[0], 2, length_img, length_img))
    count = 0
    for i in range(0, id_train):
        for j in range(0, int(run_steps[i])):
            x_train_new[count*len_img_sq: (count+1)*len_img_sq, 0] = x_train[count]
            x_train_new[count*len_img_sq: (count+1)*len_img_sq, 1:3] = coord
            images_train_new[count*len_img_sq: (count+1)*len_img_sq] = images_train[count]
            count = count + 1

    y_test_new = y_test.flatten()[:, None]
    x_test_new = np.zeros((y_test_new.shape[0], 3))
    images_test_new = np.zeros((y_test_new.shape[0], 2, length_img, length_img))
    count = 0
    for i in range(id_train, id_train+id_test):
        for j in range(0, int(run_steps[i])):
            x_test_new[count*len_img_sq: (count+1)*len_img_sq, 0] = x_test[count]
            x_test_new[count*len_img_sq: (count+1)*len_img_sq, 1:3] = coord
            images_test_new[count*len_img_sq: (count+1)*len_img_sq] = images_test[count]
            count = count + 1

    ###########################################################################
    ## Normalize
    y_train_new = y_train_new*10
    y_test_new = y_test_new*10

    return images_train_new, x_train_new, y_train_new, images_test_new, x_test_new, y_test_new, run_steps
