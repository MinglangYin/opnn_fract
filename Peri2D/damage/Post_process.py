"""
Author: Somdatta Goswami, somdatta_goswami@brown.edu
Postprocessing data.
"""

import pandas as pd
import os
import numpy as np
import scipy.io


entries = os.listdir('test_BC_center/')

originalDir = os.getcwd()
os.chdir(os.path.join(originalDir, './test_BC_center' + '/'))
originalDir = os.getcwd()

x_tr = np.zeros((64000,len(entries)),dtype = np.float32)
y_tr = np.zeros((64000,len(entries)),dtype = np.float32)
damage = np.zeros((64000,len(entries)),dtype = np.float32)

for iFile in range(len(entries)):
    os.chdir(os.path.join(originalDir, './test_' + str(iFile) + '/'))
    df = pd.read_csv('final_terror.csv', sep = ' ')
    arr = df.to_numpy()
    x_tr[:,iFile:iFile+1] = arr[:,1:2]
    y_tr[:,iFile:iFile+1] = arr[:,2:3]
    damage[:,iFile:iFile+1] = arr[:,3:4]
    os.chdir(originalDir)

os.chdir('..')
data = scipy.io.loadmat('dataset_branch.mat')
x_train = data['x_train']
y_train = data['y_train']
boundary_train = data['boundary_train']

scipy.io.savemat("dataset.mat",{'x_train':x_train, 'y_train': y_train, 'boundary_train': boundary_train, 'x_tr':x_tr, 'y_tr':y_tr, 'damage':damage})
data1 = scipy.io.loadmat('dataset.mat')