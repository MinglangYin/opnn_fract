"""
Author: Minglang Yin, minglang_yin@brown.edu
Multifidelity Architecture 
"""
import torch
import torch.nn as nn

E = 210.0*1e3
nu = 0.3
mu = 0.5*E/(1 + nu)
lmd = nu*E/((1 - 2*nu)*(1 + nu))
gc = 2.7
l = 0.0125

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
    
    def forward(self, appDisp, x):
        u_br = self._branch(appDisp)
        u_tr = self._trunk(x)
        y_out = self._out_layer(u_br*u_tr)
        u_lift = (1 - x[:, 0])*x[:, 0]*y_out[:, 0]
        v_lift = x[:, 1]*(x[:, 1] - 1)*y_out[:, 1] + x[:, 1]*appDisp[:, 0]
        phi_lift = y_out[:, 2]
        return torch.cat([u_lift[:, None], v_lift[:, None], phi_lift[:, None]], axis=1)
    
    def loss_energy(self, appDisp, x, hist_init):
        ## Total energy = elastic energy + fracture energy 
        E_e, E_c = self._get_elas_strain_energy(appDisp, x, hist_init)
        E = (E_e + E_c).mean()
        return E
    
    def _get_elas_strain_energy(self, appDisp, x, hist_init):

        y = self.forward(appDisp, x)
        u, v, phi = y[:, 0], y[:, 1], y[:, 2]

        g = (1 - phi)**2
        # get gradients
        u_g = gradients(u, x)[0]
        v_g = gradients(v, x)[0]
        phi_g = gradients(phi, x)[0]

        u_x, u_y = u_g[:, 0], u_g[:, 1]
        v_x, v_y = v_g[:, 0], v_g[:, 1]
        phi_x, phi_y = phi_g[:, 0], phi_g[:, 1]
        eps_12 = (u_y + v_x)/2

        sig_11 = self.c11*u_x + self.c12*v_y
        sig_22 = self.c21*u_x + self.c22*v_y
        sig_12 = 2*self.c33*eps_12

        # eigenvalues
        lmd_1 = (u_x + u_y)/2 + torch.sqrt( (u_x - u_y)**2 + 4*(eps_12)**2 )
        lmd_2 = (u_x + u_y)/2 - torch.sqrt( (u_x - u_y)**2 + 4*(eps_12)**2 )     
        lmd_sum = lmd_1 + lmd_2

        # decomposed tensile strain energy
        psi_pos = 0.125*lmd*(lmd_sum + torch.abs(lmd_sum))**2 \
            + 0.25*mu*( (lmd_1**2 + torch.abs(lmd_1))**2 + (lmd_2 + torch.abs(lmd_2))**2 ) 
        hist = torch.max(hist_init[:,0], psi_pos)

        E_e = 0.5*g*(sig_11*u_x + sig_22*v_y + sig_12*(u_y + v_x) )
        E_c = 0.5*gc*(phi**2/l + l*(phi_x**2 + phi_y**2)) + g*hist
        # print(f"E_e: {E_e.shape}")
        # print(f"E_c: {E_c.shape}")
        return E_e, E_c
                                                  