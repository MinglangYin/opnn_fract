"""
Author: Minglang Yin, minglang_yin@brown.edu
Multifidelity Architecture 
"""
import torch
import torch.nn as nn
#### FNN
class opnn(nn.Module):
    def __init__(self, branch_dim, trunk_dim):
        super(opnn, self).__init__()
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
        self._out_layer = nn.Linear(self.z_dim, 1, bias = True)

    def forward(self, u, x):
        y_br = self._branch(u)
        y_tr = self._trunk(x)
        y_out = self._out_layer(y_br*y_tr)
        return y_out

###### CNN
#class opnn(nn.Module):
#    def __init__(self, branch_dim, trunk_dim):
#        super(opnn, self).__init__()
#
#        self.branch_dim = branch_dim
#        self.trunk_dim = trunk_dim
#        self.z_dim = trunk_dim[-1]
#        self.ll_x = 1
#        self.ll_y = 1
#
#        ## build branch net
#        modules = []
#        for i, h_dim in enumerate(branch_dim):
#            if i == 0:
#                in_channels = h_dim
#            else:
#                modules.append(
#                    nn.Sequential(
#                        nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 2, stride= 2, padding = 0),
#                        nn.LeakyReLU(),
#                        nn.BatchNorm2d(h_dim),
#                    )
#                )
#                in_channels = h_dim
#        self._cnn = nn.Sequential(*modules)
#
#        self._flatten = nn.Sequential()
#        self._flatten.add_module('linear_1', nn.Linear(branch_dim[-1]*self.ll_x*self.ll_y, self.z_dim))
#        self._flatten.add_module('tanh_1', nn.Tanh())
#        self._flatten.add_module('linear_2', nn.Linear(self.z_dim, self.z_dim))
#        self._flatten.add_module('tanh_2', nn.Tanh())
#
#        ## build trunk net
#        modules = []
#        for i, h_dim in enumerate(trunk_dim):
#            if i == 0:
#                in_channels = h_dim
#            else:
#                modules.append(
#                    nn.Sequential(
#                        nn.Linear(in_channels, h_dim),
#                        nn.Tanh()
#                    )
#                )
#                in_channels = h_dim
#        self._trunk = nn.Sequential(*modules)
#
#        self._out_layer = nn.Linear(self.z_dim, 1, bias = False)
#
#    def trunk_net(self, x):
#        return self._trunk(x)
#
#    def branch_net(self, img):
#        y = self._cnn(img)
#        # print(y.shape)
#        y_flat = y.view(-1, self.branch_dim[-1]*self.ll_x*self.ll_y)
#        y = self._flatten(y_flat)
#        return y
#
#    def forward(self, img, x):
#        y_br = self.branch_net(img)
#        y_tr = self.trunk_net(x)
#        y_out = self._out_layer(y_br*y_tr)
#        return y_out