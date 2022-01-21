#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


torch.cuda.set_device(0)


# In[3]:


torch.set_default_tensor_type('torch.DoubleTensor')


# In[4]:


# defination of activation function
def activation(x):
    return x * torch.sigmoid(x) 


# In[5]:


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
        output = activation(self.layer_2(activation(self.layer_1(output)))) 
        output = self.layer_out(output)
        return output


# In[6]:


input_size = 1
width = 4


# In[7]:


# exact solution
def u_ex(x):  
    return torch.sin(pi*x)


# In[8]:


# f(x)
def f(x):
    return pi**2 * torch.sin(pi*x)


# In[9]:


grid_num = 200
x = torch.zeros(grid_num + 1, input_size)
for index in range(grid_num + 1):
    x[index] = index * 1 / grid_num


# In[10]:


CUDA = torch.cuda.is_available()
# print('CUDA is: ', CUDA)
if CUDA:
    net = Net(input_size,width).cuda()
    x = x.cuda()
else:
    net = Net(input_size,width)
    x = x


# In[11]:


def model(x):
    return x * (x - 1.0) * net(x)


# In[12]:


optimizer = optim.Adam(net.parameters())


# In[13]:


# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


# In[14]:


initialize_weights(net)


# In[15]:


# loss function to DGM by auto differential
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
            grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
        else:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
            grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
            
        sum_1 += ((grad_grad_x_temp[0])[0] + f(x_temp)[0])**2
    
    for index in range(1, grid_num):
        x_temp = x[index]
        x_temp.requires_grad = True
        if CUDA:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
            grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
        else:
            grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
            grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        
        sum_2 += ((grad_grad_x_temp[0])[0] + f(x_temp)[0])**2
    
    x_temp = x[0]
    x_temp.requires_grad = True
    if CUDA:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
        grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
    else:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        
    sum_a = ((grad_grad_x_temp[0])[0] + f(x_temp)[0])**2
    
    x_temp = x[grid_num]
    x_temp.requires_grad = True
    if CUDA:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
        grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape).cuda(), create_graph = True)
    else:
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        grad_grad_x_temp = torch.autograd.grad(outputs = grad_x_temp[0], inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
    
    sum_b = ((grad_grad_x_temp[0])[0] + f(x_temp)[0])**2
    
    sum_0 = h / 6 * (sum_a + 4 * sum_1 + 2 * sum_2 + sum_b)
    return sum_0


# In[16]:


def error_function(x):
    error = 0.0
    for index in range(len(x)):
        x_temp = x[index]
        error += (model(x_temp)[0] - u_ex(x_temp)[0])**2
    return error / len(x)


# In[17]:


param_num = sum(x.numel() for x in net.parameters())


# In[18]:


epoch = 10000
loss_record = np.zeros(epoch)
error_record = np.zeros(epoch)
time_start = time.time()
for i in range(epoch):
    optimizer.zero_grad()
    loss = loss_function(x)
    loss_record[i] = float(loss)
    error = error_function(x)
    error_record[i] = float(error)
    print("current epoch is: ", i)
    print("current DGM loss is: ", loss.detach())
    print("current error is: ", error.detach())
    loss.backward()
    optimizer.step() 
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')
# save model parameters
torch.save(net.state_dict(), 'net_params_DGM.pkl')


# In[19]:


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


# In[20]:


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


# In[21]:


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


# In[22]:


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]


# In[23]:


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in states.items()]


# In[24]:


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


# In[25]:


def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


# In[26]:


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


# In[27]:


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


# In[28]:


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


# In[29]:


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)


# In[30]:


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


# In[31]:


# load model parameters
pretrained_dict = torch.load('net_params_DGM.pkl')

# get state_dict
net_state_dict = net.state_dict()

# remove keys that does not belong to net_state_dict
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

# update dict
net_state_dict.update(pretrained_dict_1)

# set new dict back to net
net.load_state_dict(net_state_dict)


# In[32]:


# loss_function(x)


# In[33]:


# error_function(x)


# In[34]:


weights_temp = get_weights(net)
states_temp = net.state_dict()


# In[35]:


# print(weights_temp)


# In[36]:


step_size = 0.001  
grid = np.arange(-0.01, 0.01 + step_size, step_size)
loss_matrix = np.zeros((len(grid), len(grid)))
time_start = time.time()
for dx in grid:
    for dy in grid:
        itemindex_1 = np.argwhere(grid == dx)
        itemindex_2 = np.argwhere(grid == dy)
        
        weights = weights_temp
        states = states_temp 
        step = [dx, dy]
        direction_1 = create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter')
        normalize_directions_for_states(direction_1, states, norm='filter', ignore='ignore')
        direction_2 = create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter')
        normalize_directions_for_states(direction_2, states, norm='filter', ignore='ignore')
        directions = [direction_1, direction_2]
        set_states(net, states, directions, step)
        loss_temp = loss_function(x)
        loss_matrix[itemindex_1[0][0], itemindex_2[0][0]] = loss_temp
        
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
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')               
np.save("loss_matrix_DGM.npy",loss_matrix)


# In[37]:


# print(loss_matrix)


# In[38]:


print(loss_matrix[10, 10])


# In[ ]:





# In[ ]:




