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


class Conv2d(Module):
    ### Convolutional layer, works in the same manner as torch.nn.Conv2d : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), bias = True, dilation=(1,1) ,padding = (0,0), stride = (1,1)):
        ## instantiate parameters
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.dilation=dilation; self.padding = padding; self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels

        # compute k for the uniform distribution
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])

        # initiate the weights of the module, the values are sampled from an uniform distribution between -sqrt(k) and sqrt(k)
        self.w = empty(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(self.w.shape).zero_()

        # if needed, initiate the bias of the module, the values are sampled from an uniform distribution between -sqrt(k) and sqrt(k)
        if(bias):
            self.b = empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))
            self.db = empty(self.b.shape).zero_()
 
    def forward(self, input):
        ## forward pass of the module
        self.input = input
        self.unfolded = unfold(input, kernel_size = self.kernel_size, dilation=self.dilation, padding = self.padding, stride = self.stride)
        # the convolution is performed as a matrix multiplication after unfolding the input and reshaping the weights
        out = self.w.view(self.out_channels, -1) @ self.unfolded 

        # if bias = true, add the bias after reshaping accordingly
        if(self.bias): 
            out.add(self.b.view(1,-1,1)) 

        # compute the output dimensions
        outDim2 = (input.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] -1) - 1 )//self.stride[0] + 1
        outDim3 = (input.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] -1) - 1 )//self.stride[1] + 1

        # return the output after reshaping in the correct dimensions
        return out.view(input.shape[0], self.out_channels, outDim2 , outDim3)

    def backward(self, gradwrtoutput):
        ## backward pass of the module
        # the gradient of the output is reshaped to be able to perform a convolution using matrix multiplication
        grad_reshape = gradwrtoutput.reshape(gradwrtoutput.shape[0],self.out_channels,self.unfolded.shape[2]).transpose(1,2)

        # the gradient of the weight is computed through a convolution between the output gradient and the input
        # the result is summed and reshaped according to the weight dimensions
        self.dw = (self.unfolded @ grad_reshape).sum(0).t().view(self.dw.shape)

        # if bias = true, get the gradient of the bias by summing the gradient of the output along the first, third and fourth dimension
        if(self.bias): 
            self.db = gradwrtoutput.sum((0,2,3))
        
        # return the gradient of the input by performing a convolution between the weights and the gradient of the ouput
        # the result is reshaped and folded to return the correct dimension
        gw = (grad_reshape@self.w.view(self.out_channels, -1)).transpose(1,2)
        return fold(gw, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride = self.stride)

    def param(self):
        ## return the parameters values and gradients by pairs
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]
            
class TransposeConv2d(Module):
    ### Transposed convolutional layer, works in the same manner as torch.nn.ConvTranspose2d : https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), dilation=(1,1), padding = (0,0), bias = True):
        ## instantiate parameters
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.dilation=dilation; self.padding = padding; self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels

        # compute the "real" padding applied to the input using the padding and kernel size
        self.real_padding = (dilation[0]*(kernel_size[0] - 1) - padding[0], dilation[1]*(kernel_size[1] - 1) - padding[1])

        # compute k for the uniform distribution
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])

        # initiate the weights of the module, the values are sampled from an uniform distribution between -sqrt(k) and sqrt(k)
        self.w = empty(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(self.w.shape).zero_()

        # if needed, initiate the bias of the module, the values are sampled from an uniform distribution between -sqrt(k) and sqrt(k)
        if(bias):
            self.b = empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))
            self.db = empty(self.b.shape).zero_()
 
    def forward(self, input):
        ## forward pass of the module
        self.input = input

        # insert zeros between the columns and rows the input according to the stride
        zeroInsertDim2 = self.stride[0]*input.shape[2]-1 if(self.stride[0]>1)  else input.shape[2]
        zeroInsertDim3 = self.stride[1]*input.shape[3]-1 if(self.stride[1]>1)  else input.shape[3]
        inputZeroInserted = empty(input.shape[0], input.shape[1], zeroInsertDim3, zeroInsertDim2).zero_()
        inputZeroInserted[:,:,::self.stride[0], ::self.stride[1]] = self.input

        # the convolution is performed as a matrix multiplication after unfolding the input and reshaping the weights
        self.unfolded = unfold(inputZeroInserted, kernel_size = self.kernel_size, dilation=self.dilation, padding = self.real_padding)
        out = self.w.view(self.out_channels, -1) @ self.unfolded

        # if bias = true, add the bias after reshaping accordingly
        if(self.bias):
            out.add(self.b.view(1,-1,1))

        # compute the output dimensions
        outDim2 = (input.shape[2]-1)*self.stride[0] - 2*self.padding[0] + self.dilation[0]*(self.kernel_size[0] - 1) + 1  
        outDim3 = (input.shape[3]-1)*self.stride[1]  - 2*self.padding[1] + self.dilation[1]*(self.kernel_size[1] - 1) + 1

        # return the output after reshaping in the correct dimensions
        return out.view(input.shape[0], self.out_channels, outDim2 , outDim3)

    def backward(self, gradwrtoutput):
        ## backward pass of the module
        # the gradient of the output is reshaped to be able to perform a convolution using matrix multiplication
        grad_reshape = gradwrtoutput.reshape(gradwrtoutput.shape[0],self.out_channels,self.unfolded.shape[2]).transpose(1,2)

        # the gradient of the weight is computed through a convolution between the output gradient and the input
        # the result is summed and reshaped according to the weight dimensions
        self.dw = (self.unfolded @ grad_reshape).sum(0).t().view(self.dw.shape)

        # if bias = true, get the gradient of the bias by summing the gradient of the output along the first, third and fourth dimension
        if(self.bias):
            self.db = gradwrtoutput.sum((0,2,3))

        # return the gradient of the input by performing a convolution between the weights and the gradient of the ouput
        # the result is reshaped and folded to return the correct dimension
        gw = (grad_reshape@self.w.view(self.out_channels, -1)).transpose(1,2)
        return fold(gw, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.real_padding)

    def param(self):
        ## return the parameters values and gradients by pairs
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]


class ReLU(Module):
    ### Module of the ReLU activation function
    def forward (self, input):
        ## forward pass
        self.input = input
        zero_tensor = empty(input.shape).zero_()
        return input.maximum(zero_tensor)

    def backward(self, gradwrtoutput):
        ## backward pass
        dsigma = empty(self.input.shape)
        dsigma[self.input > 0] = 1
        return dsigma * gradwrtoutput

class Sigmoid(Module):
    ### Module of the Sigmoid activation function
    def forward (self,input):
        ## forward pass
        self.input = input
        return 1/(1+(-input).exp_())

    def backward(self, gradwrtoutput):
        ## backward pass
        sigma = 1/(1+(-self.input).exp_())
        dsigma = sigma*(1-sigma)
        return dsigma * gradwrtoutput

class Sequential(Module):
    ### Container similar to torch.nn.sequential to create a model
    def __init__(self, modules):
        ## instantiate modules
        super().__init__()
        self.modules = modules

    def forward(self, input):
        ## forward pass of the model
        self.input = input
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, gradwrtoutput):
        ## backward pass of the model
        for module in reversed(self.modules):
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput


class MSE(Module):
    ### Mean Squared Error loss function 
    def forward(self, input, target):
        ## forward pass, compute the loss
        self.input = input
        self.target = target
        return (input - target).pow(2).sum(0)/(input.shape[1]*input.shape[2]*input.shape[3])

    def backward(self):
        ## backward pass, compute the gradient of the loss
        return 2*(self.input-self.target)/(self.input.shape[1]*self.input.shape[2]*self.input.shape[3])


class SGD():
    ### Stochastic Gradient Descent
    def __init__(self, param, eta = 0.01):
        ## instantiate parameters
        self.param = param
        self.eta = eta

    def step(self):
        ## perform a step of the SGD, update parameters according to their gradient
        for param in self.param:
            val, grad = param
            val.add(-self.eta*grad)

model = Sequential([Conv2d(3, 25), ReLU(), TransposeConv2d(25, 3), Sigmoid()])
criterion = MSE()

n = 1000
center = 0.5
radius = 1 / math.sqrt((2 * math.pi))

random_tensor = empty((100, 3, 32, 32)).uniform_(0, 1)
radius_sq = math.pow(radius, 2)

temp_tensor = random_tensor.sub(center).pow(2)
target_tensor = empty(temp_tensor.shape).zero_()
target_tensor = target_tensor.where(temp_tensor < radius_sq, empty(temp_tensor.shape).zero_()+1)


output = model.forward(random_tensor)
            
loss = criterion.forward(output, target_tensor)

loss_grad = criterion.backward()
model.backward(loss_grad)
optimizer = SGD(model.param())
optimizer.step()