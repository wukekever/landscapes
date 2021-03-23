import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time

torch.set_default_tensor_type('torch.DoubleTensor')

# activation function
def activation(x):
    return x * torch.sigmoid(x) 

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

input_size = 1
width = 4
net = Net(input_size,width)



# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
            
initialize_weights(net)
            

def model(x):
    return x * (x - 1.0) * net(x)

# exact solution
def u_ex(x):  
    return torch.sin(pi*x)

# f(x)
def f(x):
    return pi**2 * torch.sin(pi*x)

grid_num = 200
x = torch.zeros(grid_num + 1, input_size)
for index in range(grid_num + 1):
    x[index] = index * 1 / grid_num

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
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        sum_1 += (0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0])
        
    for index in range(1, grid_num):
        x_temp = x[index]
        x_temp.requires_grad = True
        grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
        sum_2 += (0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0])
    
    x_temp = x[0]
    x_temp.requires_grad = True
    grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
    sum_a = 0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0]
    
    x_temp = x[grid_num]
    x_temp.requires_grad = True
    grad_x_temp = torch.autograd.grad(outputs = model(x_temp), inputs = x_temp, grad_outputs = torch.ones(model(x_temp).shape), create_graph = True)
    sum_a = 0.5*grad_x_temp[0]**2 - f(x_temp)[0]*model(x_temp)[0]
    
    sum_0 = h / 6 * (sum_a + 4 * sum_1 + 2 * sum_2 + sum_b)
    return sum_0

def error_function(x):
    error = 0.0
    for index in range(len(x)):
        x_temp = x[index]
        error += (model(x_temp)[0] - u_ex(x_temp)[0])**2
    return error / len(x)


param_num = sum(x.numel() for x in net.parameters())

# set optimizer and learning rate decay
optimizer = optim.Adam(net.parameters())
scheduler = lr_scheduler.StepLR(optimizer, 2500, 0.8) # every 2500 epoch, learning rate * 0.8

epoch = 10000
loss_record = np.zeros((epoch, 100))
error_record = np.zeros((epoch, 100))
time_start = time.time()
for j in range(100):
    initialize_weights(net)
    for i in range(epoch):
        optimizer.zero_grad()
        loss = loss_function(x)
        loss_record[i, j] = float(loss)
        error = error_function(x)
        error_record[i, j] = float(error)
        print("current epoch is: ", i)
        print("current loss is: ", loss.detach())
        print("current error is: ", error.detach())

        loss.backward()
        optimizer.step() 
        scheduler.step()
        np.save("loss_of_DRM_200_decay_meanstd.npy", loss_record)
        np.save("error_of_DRM_200_decay_meanstd.npy", error_record)

np.save("loss_of_DRM_200_decay_meanstd.npy", loss_record)
np.save("error_of_DRM_200_decay_meanstd.npy", error_record)
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')











































