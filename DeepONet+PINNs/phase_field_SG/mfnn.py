"""
Author: Minglang Yin, minglang_yin@brown.edu
Multifidelity Architecture 
"""
import torch
import torch.nn as nn
import numpy as np

E = 210.0*1e3
nu = 0.3
mu = 0.5*E/(1 + nu)
lmd = nu*E/((1 - 2*nu)*(1 + nu))
gc = 2.7
l = 0.0125
crackTip = 0.5
constB = 92
#### FNN
def gradients(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)   
    
class opnn(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        super(opnn, self).__init__()
        
        self.c11 = E*(1 - nu)/((1 + nu)*(1 - 2*nu))
        self.c22 = E*(1 - nu)/((1 + nu)*(1 - 2*nu))
        self.c12 = E*nu/((1 + nu)*(1 - 2*nu))
        self.c21 = E*nu/((1 + nu)*(1 - 2*nu))
        self.c31 = 0.0
        self.c32 = 0.0
        self.c13 = 0.0
        self.c23 = 0.0
        self.c33 = 0.5*E/(1 + nu)

        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.z_dim = trunk_dim[-1]
         ## build branch net
        modules = []
        for i, h_dim in enumerate(branch_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch = nn.Sequential(*modules)

         ## build trunk net
        modules = []
        for i, h_dim in enumerate(trunk_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._trunk = nn.Sequential(*modules)
        self._out_layer = nn.Linear(self.z_dim, 3, bias = True) 
        self._pos = nn.ReLU() 
    
    def forward(self, vDelta, x):

        u_br = self._branch(vDelta)
        u_tr = self._trunk(x)
        y_out = self._out_layer(u_br*u_tr)
    
        u_lift = x[:, 0]*y_out[:, 0]
        v_lift = x[:, 1]*(x[:, 1] - 1)*y_out[:, 1] + x[:, 1]*vDelta[:,0]
        phi_lift = y_out[:, 2]
        
        return torch.cat((u_lift[:, None], v_lift[:, None], phi_lift[:, None]), dim=1)
    
    def loss_energy(self, vDelta, x):
        ## Total energy = elastic energy + fracture energy 
        E_e, E_c = self._get_elas_strain_energy(vDelta, x)
        E = (E_e + E_c).mean()

        return E
    
    def _get_elas_strain_energy(self, vDelta, x):

        y = self.forward(vDelta, x)
        u, v, phi = y[:, 0], y[:, 1], y[:, 2]
        hist_prev = x[:,2]
        # print(f"hist: {hist.shape}")
        # print(f"u: {u.shape}")
        # print(f"v: {v.shape}")
        # print(f"phi: {phi.shape}")
        # print(f"x: {x.shape}")
        # get gradients
        u_g = gradients(u, x)[0]
        v_g = gradients(v, x)[0]
        phi_g = gradients(phi, x)[0]

        u_x, u_y = u_g[:, 0], u_g[:, 1]
        v_x, v_y = v_g[:, 0], v_g[:, 1]
        phi_x, phi_y = phi_g[:, 0], phi_g[:, 1]
        u_xy = u_y + v_x
        nabla = phi_x**2 + phi_y**2
        
        M = torch.sqrt((u_x-v_y)**2 + (u_xy**2))
        lambda1 = 0.5*(u_x + v_y) + 0.5*M
        lambda2 = 0.5*(u_x + v_y) - 0.5*M
        

        # print(f"lambda1: {lambda1.shape}") 
        # print(f"lambda2: {lambda2.shape}")
        # print(f"M: {M.shape}")
        # exit()        
        eigSum = (lambda1 + lambda2)        
        psi_pos = 0.125*lmd*(eigSum + torch.abs(eigSum))**2 + \
        0.25*mu*((lambda1 + torch.abs(lambda1))**2 + (lambda2 + torch.abs(lambda2))**2) 
#        hist_prev = self.net_hist(x)
        hist_prev = x[:,2]
        hist = torch.max(hist_prev, psi_pos)        
        sigmaX = self.c11*u_x + self.c12*v_y
        sigmaY = self.c21*u_x + self.c22*v_y
        tauXY = self.c33*u_xy
        
        E_e = 0.5*(1 - phi)**2*(sigmaX*u_x + sigmaY*v_y + tauXY*u_xy)
        E_c = 0.5*gc*(phi**2/l + l*nabla) + (1 - phi)**2* hist 
        # print(f"E_c: {E_c.shape}")
        # print(f"E_e: {E_e.shape}")
        # exit(0)
        return E_e, E_c
