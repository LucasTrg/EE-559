from ctypes.wintypes import DWORD
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math

class Module(object):
    def forward (self,*input):
        raise NotImplementedError

    def backward (self,*gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class Convolution(Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), bias = True, padding = (0,0), stride = (1,1)):
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.padding = padding; self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])
        self.w = empty(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(self.w.shape).zero_()
        if(bias):
            self.b = empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))
            self.db = empty(self.b.shape).zero_()
 
    def forward(self, input):
        self.input = input
        self.unfolded = unfold(input, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        wxb = self.w.view(self.out_channels, -1) @ self.unfolded
        if(self.bias):
            wxb.add(self.b.view(1,-1,1))

        outDim2 = (input.shape[2] + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        outDim3 = (input.shape[3] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        return wxb.view(input.shape[0], self.out_channels, outDim2 , outDim3)

    def backward(self, gradwrtoutput):
        grad_reshape = gradwrtoutput.reshape(gradwrtoutput.shape[0],self.out_channels,self.unfolded.shape[2]).transpose(1,2)
        self.dw = (self.unfolded @ grad_reshape).sum(0).t().view(self.dw.shape)
        if(self.bias):
            self.db = gradwrtoutput.sum((0,2,3))
        gw = (grad_reshape@self.w.view(self.out_channels, -1)).transpose(1,2)
        return fold(gw, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride = self.stride)

    def param(self):
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]
            
class TransposedConvolution(Module):
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (0,0), bias = True):
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias 
        self.pad = padding
        self.padding = (kernel_size[0] - padding[0] -1, kernel_size[1] - padding[1] -1)
        self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])
        self.w = empty(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(self.w.shape).zero_()
        if(bias):
            self.b = empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))
            self.db = empty(self.b.shape).zero_()
 
    def forward(self, input):
        self.input = input
        zeroInsertDim2 = self.stride[0]*input.shape[2]-1 if(self.stride[0]>1)  else input.shape[2]
        zeroInsertDim3 = self.stride[1]*input.shape[3]-1 if(self.stride[1]>1)  else input.shape[3]
        inputZeroInserted = empty(input.shape[0], input.shape[1], zeroInsertDim3, zeroInsertDim2).zero_()
        inputZeroInserted[:,:,::self.stride[0], ::self.stride[1]] = self.input
        self.unfolded = unfold(inputZeroInserted, kernel_size = self.kernel_size, padding = self.padding)
        wxb = self.w.view(self.out_channels, -1) @ self.unfolded
        if(self.bias):
            wxb.add(self.b.view(1,-1,1))

        outDim2 = (input.shape[2]-1)*self.stride[0] - 2*self.pad[0] + self.kernel_size[0] 
        outDim3 = (input.shape[3]-1)*self.stride[1]  - 2*self.pad[1] + self.kernel_size[1]
        return wxb.view(input.shape[0], self.out_channels, outDim2 , outDim3)

    def backward(self, gradwrtoutput):
        grad_reshape = gradwrtoutput.reshape(gradwrtoutput.shape[0],self.out_channels,self.unfolded.shape[2]).transpose(1,2)
        self.dw = (self.unfolded @ grad_reshape).sum(0).t().view(self.dw.shape)
        if(self.bias):
            self.db = gradwrtoutput.sum((0,2,3))
        gw = (grad_reshape@self.w.view(self.out_channels, -1)).transpose(1,2)
        return fold(gw, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding)

    def param(self):
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]


class ReLu(Module):
    def forward (self, input):
        self.input = input
        zero_tensor = empty(input.shape).zero_()
        return input.maximum(zero_tensor)

    def backward(self, gradwrtoutput):
        dsigma = empty(self.input.shape)
        dsigma[self.input > 0] = 1
        return dsigma * gradwrtoutput

class Sigmoid(Module):
    def forward (self,input):
        self.input = input
        return 1/(1+(-input).exp_())

    def backward(self, gradwrtoutput):
        sigma = 1/(1+(-input).exp_())
        dsigma = sigma*(1-sigma)
        return dsigma * gradwrtoutput

class Sequential(Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, input):
        self.input = input
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, gradwrtoutput):
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput


class MSELoss(Module):
    def forward(self, input, target):
        self.input = input
        self.target = target
        return sum((input - target).pow(2).sum(0))/(input.shape[1]*input.shape[2]*input.shape[3])

    def backward(self):
        return 2*(self.input-self.target)/(self.input.shape[1]*self.input.shape[2]*self.input.shape[3])


class SGDOptimizer():
    def __init__(self, param, eta = 0.01):
        self.param = param
        self.eta = eta

    def step(self):
        for param in self.param:
            val, grad = param
            val.add(-self.eta*grad)

model = Sequential([Convolution(3, 25), ReLu(), TransposedConvolution(25, 3)])
criterion = MSELoss()

n = 1000
center = 0.5
radius = 1 / math.sqrt((2 * math.pi))

random_tensor = empty((100, 3, 32, 32)).uniform_(0, 1)
radius_sq = math.pow(radius, 2)

temp_tensor = random_tensor.sub(center).pow(2)
target_tensor = empty(temp_tensor.shape).zero_()
target_tensor = target_tensor.where(temp_tensor < radius_sq, empty(temp_tensor.shape).zero_()+1)

log_losses = []
mean_losses = 0

output = model.forward(random_tensor)
            
loss = criterion.forward(output, target_tensor)

loss_grad = criterion.backward()
model.backward(loss_grad)
optimizer = SGDOptimizer(model.param())
optimizer.step()