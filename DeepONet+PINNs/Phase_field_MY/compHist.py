"""
Author: Minglang Yin, minglang_yin@brown.edu, Somdatta Goswami, somdatta_goswami@brown.edu
Script for computing the history function
"""
import numpy as np
import scipy.io as io
from scipy.io import savemat
import matplotlib.pyplot as plt

E = 210.0*1e3
nu = 0.3
mu = 0.5*E/(1 + nu)
lmd = nu*E/((1 - 2*nu)*(1 + nu))
gc = 2.7
l = 0.0125
crackTip = 0.5
constB = 92

dataset = io.loadmat('dataset.mat')
x = dataset['coord']
u = dataset['u_train']
v = dataset['v_train']

num_cols = 122
num_rows = 122
x_reshape= x[:,0].reshape(num_rows, num_cols)
y_reshape= x[:,1].reshape(num_rows, num_cols)
u_reshape= u[:,0].reshape(num_rows, num_cols)

dx = np.gradient(x_reshape)[1]
dx = dx.reshape(num_rows*num_cols,1)

dy = np.gradient(y_reshape)[0]
dy = dy.reshape(num_rows*num_cols,1)
u_x = np.zeros((14884, 8))
u_y = np.zeros((14884, 8))
v_x = np.zeros((14884, 8))
v_y = np.zeros((14884, 8))

for i in range(0,8):
    
    u_reshape= u[:,i:i+1].reshape(num_rows, num_cols)
    u_g_x = np.gradient(u_reshape)[0]
    u_g_y = np.gradient(u_reshape)[1]
    u_g_x = u_g_x.reshape(num_rows*num_cols,1)
    u_g_y = u_g_y.reshape(num_rows*num_cols,1)
    u_g_x = u_g_x/dx
    u_x[:,i:i+1] = u_g_x
    u_g_y = u_g_y/dy
    u_y[:,i:i+1] = u_g_y
    
    v_reshape= v[:,i:i+1].reshape(num_rows, num_cols)
    v_g_x = np.gradient(v_reshape)[0]
    v_g_y = np.gradient(v_reshape)[1]
    v_g_x = v_g_x.reshape(num_rows*num_cols,1)
    v_g_y = v_g_y.reshape(num_rows*num_cols,1)
    v_g_x = v_g_x/dx
    v_x[:,i:i+1] = v_g_x
    v_g_y = v_g_y/dy
    v_y[:,i:i+1] = v_g_y

u_xy = u_y + v_x
M = np.sqrt((u_x-v_y)**2 + (u_xy**2))
lambda1 = 0.5*(u_x + v_y) + 0.5*M
lambda2 = 0.5*(u_x + v_y) - 0.5*M
eigSum = (lambda1 + lambda2) 
strain_pos = 0.125*lmd*(eigSum + np.abs(eigSum))**2 + \
    0.25*mu*((lambda1 + np.abs(lambda1))**2 + (lambda2 + np.abs(lambda2))**2)
    
fig = plt.figure(figsize=(5, 5))
plt.scatter(x[:,0], x[:,1], c=strain_pos[:,5])
plt.savefig('strainPos.png')

init_hist = np.zeros((x.shape[0], 1))
dist = np.where(x[:, 0] > crackTip, np.sqrt((x[:, 0] - 0.5)**2 + (x[:, 1] - 0.5)**2), np.abs(x[:, 1] - 0.5))
init_hist[:,0] = np.where(dist <= 0.5*l, constB*gc*0.5*(1 - (2*dist/l))/l, init_hist[:,0])

fig = plt.figure(figsize=(5, 5))
plt.scatter(x[:,0], x[:,1], c=init_hist[:,0])
plt.savefig('History.png')

hist = np.zeros((14884, 8))
for iStep in range(0,8):
    hist[:,iStep] = np.maximum(init_hist[:,0], strain_pos[:,iStep])

fig = plt.figure(figsize=(5, 5))
plt.scatter(x[:,0], x[:,1], c=hist[:,7])
plt.savefig('History_1.png')

savemat('history.mat', {'hist':hist})

