#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
# import matplotlib.pyplot as plt
from math import *
import time


# In[2]:


torch.cuda.set_device(5)
torch.set_default_tensor_type('torch.DoubleTensor')


# In[3]:


# activation function
def activation(x):
    return x * torch.sigmoid(x)


# In[4]:


# build ResNet with three blocks
class Net(torch.nn.Module):
    def __init__(self,input_width,layer_width):
        super(Net,self).__init__()
        self.layer_in = torch.nn.Linear(input_width, layer_width)
        self.layer1 = torch.nn.Linear(layer_width, layer_width)
        self.layer2 = torch.nn.Linear(layer_width, layer_width)
        self.layer3 = torch.nn.Linear(layer_width, layer_width)
        self.layer4 = torch.nn.Linear(layer_width, layer_width)
        self.layer5 = torch.nn.Linear(layer_width, layer_width)
        self.layer6 = torch.nn.Linear(layer_width, layer_width)
        self.layer_out = torch.nn.Linear(layer_width, 1)
    def forward(self,x):
        y = self.layer_in(x)
        y = y + activation(self.layer2(activation(self.layer1(y)))) # residual block 1
        y = y + activation(self.layer4(activation(self.layer3(y)))) # residual block 2
        y = y + activation(self.layer6(activation(self.layer5(y)))) # residual block 3
        output = self.layer_out(y)
        return output


# In[5]:


dimension = 10


# In[6]:


input_width,layer_width = dimension, 20


# In[7]:


net = Net(input_width,layer_width).cuda() # network for u on gpu


# In[8]:


# defination of exact solution
def u_ex(x):     
    temp = 1.0
    for i in range(dimension):
        temp = temp * torch.sin(pi*x[:, i])
    u_temp = 1.0 * temp
    return u_temp.reshape([x.size()[0], 1])


# In[9]:


# defination of f(x)
def f(x):
    temp = 1.0
    for i in range(dimension):
        temp = temp * torch.sin(pi*x[:, i])
    u_temp = 1.0 * temp
    f_temp = dimension * pi**2 * u_temp 
    return f_temp.reshape([x.size()[0],1])


# In[10]:


# generate points by random
def generate_sample(data_size):
    sample_temp = torch.rand(data_size, dimension)
    return sample_temp.cuda()


# In[11]:


def model(x):
    x_temp = x.cuda()
    D_x_0 = torch.prod(x_temp, axis = 1).reshape([x.size()[0], 1]) 
    D_x_1 = torch.prod(1.0 - x_temp, axis = 1).reshape([x.size()[0], 1]) 
    model_u_temp = D_x_0 * D_x_1 * net(x)
    return model_u_temp.reshape([x.size()[0], 1])


# In[12]:


# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


# In[13]:


initialize_weights(net)


# In[14]:


# loss function to DRM by auto differential
def loss_function(x):
#     x = generate_sample(data_size).cuda()
#     x.requires_grad = True
    u_hat = model(x)
    grad_u_hat = torch.autograd.grad(outputs = u_hat, inputs = x, grad_outputs = torch.ones(u_hat.shape).cuda(), create_graph = True)
    grad_u_sq = ((grad_u_hat[0]**2).sum(1)).reshape([len(grad_u_hat[0]), 1])
    part = torch.sum(0.5 * grad_u_sq  - f(x) * u_hat)  / len(x)
    return part


# In[15]:


def relative_l2_error():
    data_size_temp = 1000
    x = generate_sample(data_size_temp).cuda() 
    predict = model(x)
    exact = u_ex(x)
    value = torch.sqrt(torch.sum((predict - exact)**2))/torch.sqrt(torch.sum((exact)**2))
    return value


# In[16]:


optimizer = optim.Adam(net.parameters())


# In[ ]:


epoch = 50000
data_size = 100000
loss_record = np.zeros(epoch)
error_record = np.zeros(epoch)
time_start = time.time()
for i in range(epoch):
    optimizer.zero_grad()
    x = generate_sample(data_size).cuda()
    x.requires_grad = True
    loss = loss_function(x)
    loss_record[i] = float(loss)
    error = relative_l2_error()
    error_record[i] = float(error)
#     np.save("DRM_loss_10d.npy", loss_record)
#     np.save("DRM_error_10d.npy", error_record)
    if i % 5 == 0:
        print("current epoch is: ", i)
        print("current loss is: ", loss.detach())
        print("current error is: ", error.detach())
    if i == epoch - 1:
        # save model parameters
        torch.save(net.state_dict(), 'net_params_DRM.pkl')
        
    loss.backward()
    optimizer.step() 
    torch.cuda.empty_cache() # clear memory
    
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')


# In[ ]:


np.save("DRM_loss_10d.npy", loss_record)
np.save("DRM_error_10d.npy", error_record)


# In[ ]:





# In[ ]:





# In[ ]:




