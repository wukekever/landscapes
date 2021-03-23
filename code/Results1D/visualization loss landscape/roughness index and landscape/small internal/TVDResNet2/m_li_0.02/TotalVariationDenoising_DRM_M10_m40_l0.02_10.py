#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from math import *
import time

import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR


# In[ ]:


torch.cuda.set_device(4)


# In[ ]:


torch.set_default_tensor_type('torch.DoubleTensor')


# In[ ]:


# defination of activation function
def activation(x):
    return x * torch.sigmoid(x) 


# In[ ]:


# build ResNet with one blocks
class Net(nn.Module):
    def __init__(self,input_size,width):
        super(Net,self).__init__()
        self.layer_in = nn.Linear(input_size,width)
        self.layer_1 = nn.Linear(width,width)
        self.layer_2 = nn.Linear(width,width)
        self.layer_out = nn.Linear(width,1)
    def forward(self,x):
        output = self.layer_in(x)
        output = output + activation(self.layer_2(activation(self.layer_1(output)))) # residual block 1
        output = self.layer_out(output)
        return output


# In[ ]:


input_size = 1
width = 4


# In[ ]:


# exact solution
def u_ex(x):  
    return torch.sin(pi*x)


# In[ ]:


# f(x)
def f(x):
    return pi**2 * torch.sin(pi*x)


# In[ ]:


grid_num = 200
x = torch.zeros(grid_num + 1, input_size)
for index in range(grid_num + 1):
    x[index] = index * 1 / grid_num


# In[ ]:


CUDA = torch.cuda.is_available()
# print('CUDA is: ', CUDA)
if CUDA:
    net = Net(input_size,width).cuda()
    x = x.cuda()
else:
    net = Net(input_size,width)
    x = x


# In[ ]:


def model(x):
    return x * (x - 1.0) * net(x)


# In[ ]:


optimizer = optim.Adam(net.parameters())


# In[ ]:


# loss function to DRM by auto differential
def loss_function(x):
    h = 1 / grid_num
    sum_0 = 0.0
    sum_1 = 0.0
    sum_2 = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for index in range(grid_num):
        x_temp = x[index] + h / 2 
        x_temp.requires_grad = True
        if CUDA:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
        else:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        sum_1 += (0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0])
        
    for index in range(1, grid_num):
        x_temp = x[index]
        x_temp.requires_grad = True
        if CUDA:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
        else:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        sum_2 += (0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0])
    
    x_temp = x[0]
    x_temp.requires_grad = True
    if CUDA:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
    else:  
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
    sum_a = 0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0]
    
    x_temp = x[grid_num]
    x_temp.requires_grad = True
    if CUDA:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
    else:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
    
    sum_a = 0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0]
    
    sum_0 = h / 6 * (sum_a + 4 * sum_1 + 2 * sum_2 + sum_b)
    return sum_0


# In[ ]:


param_num = sum(x.numel() for x in net.parameters())


# In[ ]:


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


# In[ ]:


def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))


# In[ ]:


def set_states(net, states, directions=None, step=None):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    if directions is None:
        net.load_state_dict(states)
    else:
        assert step is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        net.load_state_dict(new_states)


# In[ ]:


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]


# In[ ]:


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in states.items()]


# In[ ]:


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


# In[ ]:


def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


# In[ ]:


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


# In[ ]:


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


# In[ ]:


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


# In[ ]:


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)


# In[ ]:


def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.
        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net) # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == 'states':
        states = net.state_dict() # a dict of parameters, including BN's running mean/var.
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction


# In[ ]:


# # load model parameters
# pretrained_dict = torch.load('net_params_DRM.pkl')

# # get state_dict
# net_state_dict = net.state_dict()

# # remove keys that does not belong to net_state_dict
# pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

# # update dict
# net_state_dict.update(pretrained_dict_1)

# # set new dict back to net
# net.load_state_dict(net_state_dict)


# In[ ]:


# weights_temp = get_weights(net)
# states_temp = net.state_dict()


# In[ ]:


def tvd(m, l_i):
    
    # load model parameters
    pretrained_dict = torch.load('net_params_DRM.pkl')
    
    # get state_dict
    net_state_dict = net.state_dict()

    # remove keys that does not belong to net_state_dict
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

    # update dict
    net_state_dict.update(pretrained_dict_1)

    # set new dict back to net
    net.load_state_dict(net_state_dict)
    
    weights_temp = get_weights(net)
    states_temp = net.state_dict()
    
    step_size = 2 * l_i / m  
    grid = np.arange(-l_i, l_i + step_size, step_size)
    num_direction = 10
    loss_matrix = torch.zeros((num_direction, len(grid)))

    for temp in range(num_direction):
        weights = weights_temp
        states = states_temp
        direction_temp = create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter')
        normalize_directions_for_states(direction_temp, states, norm='filter', ignore='ignore')
        directions = [direction_temp]

        for dx in grid:
            itemindex_1 = np.argwhere(grid == dx)
            step = dx

            set_states(net, states, directions, step)
            loss_temp = loss_function(x)
            loss_matrix[temp, itemindex_1[0]] = loss_temp

            # get state_dict
            net_state_dict = net.state_dict()
            # remove keys that does not belong to net_state_dict
            pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
            # update dict
            net_state_dict.update(pretrained_dict_1)
            # set new dict back to net
            net.load_state_dict(net_state_dict)
            weights_temp = get_weights(net)
            states_temp = net.state_dict()

    interval_length = grid[-1] - grid[0]
    TVD = 0.0
    for temp in range(num_direction):
        for index in range(loss_matrix.size()[1] - 1):
            TVD = TVD + np.abs(float(loss_matrix[temp, index] - loss_matrix[temp, index + 1]))
    TVD = TVD / interval_length / num_direction 
    
    return TVD


# In[ ]:


Mmod10 = 1
m = 40
l_i = 0.02


# In[ ]:


TVD_DRM = 0.0

time_start = time.time()

for count in range(Mmod10):
    TVD_temp = tvd(m, l_i)
    print('current TVD of DRM is: ', TVD_temp)
    TVD_DRM = TVD_DRM + TVD_temp
    print((count + 1) / Mmod10 * 100, '% finished.')

time_end = time.time()
print('Total time costs: ', time_end-time_start, 'seconds')      
    
TVD_DRM = TVD_DRM / Mmod10
print('TVD of DRM is: ', TVD_DRM)


# In[ ]:




