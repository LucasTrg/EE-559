from ctypes.wintypes import DWORD
from torch import empty
from torch.nn.functional import fold, unfold
import math

class Module(object):
    def forward (self,*input):
        raise NotImplementedError

    def backward (self,*gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
    
    def set_param(self, *new_params):
        return None


class Conv2d(Module):
    ### Convolutional layer, works in the same manner as torch.nn.Conv2d : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), bias = True, padding = (0,0), stride = (1,1)):
        ## instantiate parameters
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.padding = padding; self.stride = stride
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
        self.unfolded = unfold(input, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        # the convolution is performed as a matrix multiplication after unfolding the input and reshaping the weights
        output = self.w.view(self.out_channels, -1) @ self.unfolded 
        # if bias = true, add the bias after flattening it
        if(self.bias): 
            output+=self.b.view(1,-1,1)
        # compute the output dimensions
        Hout = (input.shape[2] + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        Wout = (input.shape[3] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        # return the output in the output dimensions
        return output.view(input.shape[0], self.out_channels, Hout , Wout)

    def backward(self, gradwrtoutput):
        ## backward pass of the module
        # the gradient of the images is reshaped to 3 dimensions
        gradwrtouput_reshaped = gradwrtoutput.view(gradwrtoutput.shape[0], gradwrtoutput.shape[1], -1)

        #Weight gradient, it is a convolution as well: dw = dy*(dx)', summed in the batch dimension
        self.dw = (gradwrtouput_reshaped @ self.unfolded.transpose(1,2)).sum(0).view(self.dw.shape)

        # if bias = true, get the gradient of the bias by summing the gradient to the channels(2nd) dimension
        if(self.bias):
            self.db = gradwrtoutput.sum((0,2,3))
        
        # Gradient with respect to the input, dx = (dw)'*dy
        gradwrtinput_unfolded = (self.w.view(self.out_channels, -1).t() @ gradwrtouput_reshaped)

        #Folded back to 4 dimensions
        return fold(gradwrtinput_unfolded, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride = self.stride)

    def param(self):
        ## return the parameters values and gradients by pairs
        if self.bias:
            self.parameters=[(self.w, self.dw), (self.b, self.db)]
        else:
            self.parameters=[(self.w, self.dw)]
        return self.parameters
    
    def set_param(self, new_params):
        self.w=new_params[0][0]
        #self.dw=new_params[0][1]
        if self.bias:
            self.b=new_params[1][0]
            #self.db=new_params[1][1]

class TransposeConv2d(Module):
    ### Transposed convolutional layer, works in the same manner as torch.nn.ConvTranspose2d : https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (0,0), bias = True):
        ## instantiate parameters
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.padding = padding; self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels

        # compute k for the uniform distribution
        k = 1/(out_channels*kernel_size[0]*kernel_size[1])

        # initiate the weights of the module, the values are sampled from an uniform distribution between -sqrt(k) and sqrt(k)
        self.w = empty(in_channels, out_channels, kernel_size[0], kernel_size[1]).uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = empty(self.w.shape).zero_()


        # if needed, initiate the bias of the module, the values are sampled from an uniform distribution between -sqrt(k) and sqrt(k)
        if(bias):
            self.b = empty(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))
            self.db = empty(self.b.shape).zero_()
 
    def forward(self, input):
        ## forward pass of the module, analog to the backward pass of Conv2D
        # the input is reshaped to columns for the matrix operations
        self.input = input
        self.input_reshaped = input.view(input.shape[0],input.shape[1],-1)

        # compute the output dimensions
        Hout = (input.shape[2]-1)*self.stride[0] - 2*self.padding[0] + self.kernel_size[0]
        Wout = (input.shape[3]-1)*self.stride[1] - 2*self.padding[1] + self.kernel_size[1]

        #(w)'*x, transposed convolution as a matrix multiplication
        output_unfolded = (self.w.view(self.in_channels, -1).t()@self.input_reshaped)

        #Fold back to 4 dimensions
        output = fold(output_unfolded, (Hout, Wout), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride = self.stride)

        if(self.bias):
            #add the bias
            output+=self.b.view(1,-1,1,1)

        return output


    def backward(self, gradwrtoutput):
        ## backward pass of the module, analog the forward pass of Conv2D
        # Unfold input
        self.unfolded = unfold(gradwrtoutput, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)

        #Weight gradient, it is a convolution as well: dw = dy*(dx)', summed in the batch dimension
        self.dw = (self.input_reshaped @ self.unfolded.transpose(1,2)).sum(0).view(self.dw.shape)
        # if bias = true, get the gradient of the bias by summing the gradient to the channels(2nd) dimension
        if(self.bias): 
            self.db = gradwrtoutput.sum((0,2,3))

        # Compute the gradient according to the input: dx = dw*x
        gradwrtinput = self.w.view(self.in_channels, -1) @ self.unfolded 

        # return the gradient with respect to the input after reshaping in the input dimensions 
        return gradwrtinput.view(gradwrtoutput.shape[0], self.in_channels, self.input.shape[2] , self.input.shape[3])

    def param(self):
        ## return the parameters values and gradients by pairs
        if (self.bias):
            self.parameters=[(self.w, self.dw), (self.b, self.db)]
        else:
            self.parameters=[(self.w, self.dw)]
        return self.parameters
    
    def set_param(self, new_params):
        self.w=new_params[0][0]
        #self.dw=new_params[0][1]
        if self.bias:
            self.b=new_params[1][0]
            #self.db=new_params[1][1]



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

    def param(self):
        
        param = []
        for module in self.modules:
            param.extend(module.param())
        return param

    def set_param(self, new_params):
        i=0
        #print('BAAAAAAAAAAAAAAAAAAAA')

        for module in self.modules:
            if module.param()!=[]:
                module.set_param(new_params[i])
            i+=1



class MSE(Module):
    ### Mean Squared Error loss function 
    def forward(self, input, target):
        ## forward pass, compute the loss
        self.input = input
        self.target = target
        return (input - target).pow(2).sum()/(input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3])

    def backward(self):
        ## backward pass, compute the gradient of the loss
        return 2*(self.input-self.target)/(self.input.shape[0]*self.input.shape[1]*self.input.shape[2]*self.input.shape[3])


class SGD():
    ### Stochastic Gradient Descent
    def __init__(self, eta = 1e-4):
        ## instantiate parameters
        self.eta = eta

    def step(self, parameters):
        ## perform a step of the SGD, update parameters according to their gradient
        for param in parameters:
            val, grad = param
            val -= self.eta*grad

"""
model = Sequential([Conv2d(3, 25), ReLU(), TransposeConv2d(25, 3)])
criterion = MSE()
optimizer = SGD()
n = 1000
center = 0.5
radius = 1 / math.sqrt((2 * math.pi))

random_tensor = empty((100, 3, 32, 32)).uniform_(0, 1)
radius_sq = math.pow(radius, 2)

temp_tensor = random_tensor.sub(center).pow(2)
target_tensor = empty(temp_tensor.shape).zero_()
target_tensor = target_tensor.where(temp_tensor < radius_sq, empty(temp_tensor.shape).zero_()+1)

for i in range(1):
    output = model.forward(random_tensor)
                
    loss = criterion.forward(output, target_tensor)
    loss_grad = criterion.backward()
    model.backward(loss_grad)
    optimizer.step(model.param())"""

