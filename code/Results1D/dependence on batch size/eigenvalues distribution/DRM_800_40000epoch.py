import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

def model(x):
    return x * (x - 1.0) * net(x)

# exact solution
def u_ex(x):  
    return torch.sin(pi*x)

# f(x)
def f(x):
    return pi**2 * torch.sin(pi*x)

grid_num = 800
x = torch.zeros(grid_num + 1, input_size)
for index in range(grid_num + 1):
    x[index] = index * 1 / grid_num

optimizer = optim.Adam(net.parameters())

# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

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

def gradient(outputs, inputs, grad_outputs = None, retain_graph = None, create_graph = False):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused = True,
                                retain_graph = retain_graph,
                                create_graph = create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])

def jacobian(outputs, inputs, create_graph = False):
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)

def hessian(output, inputs, out = None, allow_unused = False, create_graph = False):    
#     assert output.ndimension() == 0
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph = True, allow_unused = allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph = True, create_graph = create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out

param_num = sum(x.numel() for x in net.parameters())

epoch = 40000
period = 50
loss_record = np.zeros(epoch)
error_record = np.zeros(epoch)
eigen_vals_record = np.zeros((param_num, int(epoch / period)))
time_start = time.time()
for i in range(epoch):
    optimizer.zero_grad()
    loss = loss_function(x)
    loss_record[i] = float(loss)
    error = error_function(x)
    error_record[i] = float(error)
    print("current epoch is: ", i)
    print("current loss is: ", loss.detach())
    print("current error is: ", error.detach())
    if i % period == 0:
        h = hessian(loss, net.parameters())
        array_h = h.numpy() 
        eigen_vals, eigen_vecs = np.linalg.eig(array_h)
        eigen_vals_record[:, int(i / period)] = np.array(sorted(np.real(eigen_vals))[::-1]) 
    loss.backward()
    optimizer.step() 
    np.save("loss_of_DRM_800.npy", loss_record)
    np.save("error_of_DRM_800.npy", error_record)
    np.save("eigen_vals_of_DRM_800.npy", eigen_vals_record)
np.save("loss_of_DRM_800.npy", loss_record)
np.save("error_of_DRM_800.npy", error_record)
np.save("eigen_vals_of_DRM_800.npy", eigen_vals_record)
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')












































