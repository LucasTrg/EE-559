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
        #print('conv',input.shape)
        self.input = input
<<<<<<< HEAD

        self.unfolded = unfold(input, kernel_size = self.kernel_size, dilation=self.dilation, padding = self.padding, stride = self.stride)
        print('conv unfolded ',self.unfolded.shape, self.w.view(self.out_channels, -1).shape)


=======
        self.unfolded = unfold(input, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
>>>>>>> parent of 985abbb (sync)
        # the convolution is performed as a matrix multiplication after unfolding the input and reshaping the weights
        output = self.w.view(self.out_channels, -1) @ self.unfolded 
        # if bias = true, add the bias after flattening it
        if(self.bias): 
            output.add(self.b.view(1,-1,1)) 
        # compute the output dimensions
<<<<<<< HEAD
        Hout = (input.shape[2] + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        Wout = (input.shape[3] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1
        # return the output in the output dimensions
        return output.view(input.shape[0], self.out_channels, Hout , Wout)
=======
        outDim2 = (input.shape[2] + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        outDim3 = (input.shape[3] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1

        # return the output after reshaping in the correct dimensions
        return out.view(input.shape[0], self.out_channels, outDim2 , outDim3)
>>>>>>> parent of 985abbb (sync)

    def backward(self, gradwrtoutput):
        ## backward pass of the module
        # the gradient of the images is reshaped to 3 dimensions
        gradwrtouput_im2col = gradwrtoutput.view(gradwrtoutput.shape[0], gradwrtoutput.shape[1], -1)

        #Weight gradient, it is a convolution as well: dw = dy*(dx)', summed in the batch dimension
        self.dw = (gradwrtouput_im2col @ self.unfolded.transpose(1,2)).sum(0).view(self.dw.shape)

        # if bias = true, get the gradient of the bias by summing the gradient to the channels(2nd) dimension
        if(self.bias):
            self.db = gradwrtoutput.sum((0,2,3))
        
        # Gradient with respect to the input, dx = (dw)'*dy
        gradwrtinput = (self.w.view(self.out_channels, -1).t()@gradwrtouput_im2col)

        #Folded back to 4 dimensions
        return fold(gradwrtinput, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride = self.stride)

    def param(self):
        ## return the parameters values and gradients by pairs
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]

class TransposeConv2d(Module):
    ### Transposed convolutional layer, works in the same manner as torch.nn.ConvTranspose2d : https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (0,0), bias = True):
        ## instantiate parameters
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.padding = padding; self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels

        # compute the "real" padding applied to the input using the padding and kernel size
        self.real_padding = (kernel_size[0] - padding[0] -1, kernel_size[1] - padding[1] -1)

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
        self.input_im2col = input.view(input.shape[0],input.shape[1],-1)

        # compute the output dimensions
        Hout = (input.shape[2]-1)*self.stride[0] - 2*self.padding[0] + self.kernel_size[0]
        Wout = (input.shape[3]-1)*self.stride[1] - 2*self.padding[1] + self.kernel_size[1]

        #(w)'*x, transposed convolution as a matrix multiplication
        wx = (self.w.view(self.in_channels, -1).t()@self.input_im2col)

        #Fold back to 4 dimensions
        output = fold(wx, (Hout, Wout), (self.kernel_size[0], self.kernel_size[1]), padding=self.padding, stride = self.stride)

        if(self.bias):
            #add the bias
            output.add(self.b.view(1,-1,1,1));

        return output

        # insert zeros between the columns and rows the input according to the stride
        #zeroInsertDim2 = self.stride[0]*input.shape[2] if(self.stride[0]>1)  else input.shape[2]
        #zeroInsertDim3 = self.stride[1]*input.shape[3] if(self.stride[1]>1)  else input.shape[3]
        #zeroInsertDim1 = input.shape[1]//self.stride[0]#if we say that the stride is even for 0 and 1 dim

<<<<<<< HEAD
        #inputZeroInserted = empty(input.shape[0], zeroInsertDim1, zeroInsertDim2, zeroInsertDim3).zero_()
        #inputZeroInserted[:,:,::self.stride[0], ::self.stride[1]] = self.input[:,::self.stride[0],:,:]


        # compute the output dimensions
        outDim2 = (input.shape[2]-1)*self.stride[0] - 2*self.padding[0] + self.dilation[0]*(self.kernel_size[0] - 1) + 1  
        outDim3 = (input.shape[3]-1)*self.stride[1]  - 2*self.padding[1] + self.dilation[1]*(self.kernel_size[1] - 1) + 1
        output_size=[self.input.shape[0],self.out_channels, outDim2, outDim3]
        print('output size',output_size)
        

        #L3=(output_size[3]+2*self.padding[3]-self.dilation[3]*(self.kernel_size[3] - 1) -1)/stride[3] + 1
        #L2=(output_size[2]+2*self.padding[2]-self.dilation[2]*(self.kernel_size[2] - 1) -1)/stride[2] + 1


        self.input=self.input.reshape(self.input.shape[0], self.input.shape[1]*self.kernel_size[0]*self.kernel_size[1], -1)
        print(self.input.shape)
        print(self.w.shape, self.input.shape)

        
        print('zero inserted:', self.input.shape,  'wtranspose: ', self.w.view(self.in_channels, -1).t().shape)
        out = self.w.view(self.out_channels, -1) @ self.input
        print(out.shape)
=======
        # the convolution is performed as a matrix multiplication after unfolding the input and reshaping the weights
        self.unfolded = unfold(inputZeroInserted, kernel_size = self.kernel_size, padding = self.real_padding)
        out = self.w.view(self.out_channels, -1) @ self.unfolded
>>>>>>> parent of 985abbb (sync)

        # if bias = true, add the bias after reshaping accordingly
        if(self.bias):
            out.add(self.b.view(1,-1,1))


        # compute the output dimensions
        outDim2 = (input.shape[2]-1)*self.stride[0] - 2*self.padding[0] + self.kernel_size[0] 
        outDim3 = (input.shape[3]-1)*self.stride[1]  - 2*self.padding[1] + self.kernel_size[1]

        # return the output after reshaping in the correct dimensions
        self.folded=fold(self.input, (outDim2, outDim3), (self.kernel_size[0], self.kernel_size[1]), dilation= self.dilation, padding=self.real_padding, stride = self.stride)
        print(self.folded.shape)
        return self.folded
            #input.shape[0], self.out_channels, outDim2 , outDim3)

    def backward(self, gradwrtoutput):
        ## backward pass of the module, analog the forward pass of Conv2D

        # Unfold input
        self.unfolded = unfold(gradwrtoutput, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)

        # Compute the gradient according to the input: dx = dw*x
        gradwrtinput = self.w.view(self.in_channels, -1) @ self.unfolded 

        #Weight gradient, it is a convolution as well: dw = dy*(dx)', summed in the batch dimension
        self.dw = (self.input_im2col @ self.unfolded.transpose(1,2)).sum(0).view(self.dw.shape)

        # if bias = true, get the gradient of the bias by summing the gradient to the channels(2nd) dimension
        if(self.bias): 
            self.db = gradwrtoutput.sum((0,2,3))

        # compute the output dimensions
        Hout = (gradwrtoutput.shape[2] + 2*self.padding[0] - self.kernel_size[0])//self.stride[0] + 1
        Wout = (gradwrtoutput.shape[3] + 2*self.padding[1] - self.kernel_size[1])//self.stride[1] + 1

        # return the gradient with respect to the input after reshaping in the correct dimensions
        return gradwrtinput.view(gradwrtoutput.shape[0], self.in_channels, Hout , Wout)

    def param(self):
        ## return the parameters values and gradients by pairs
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]





class Nearest_Upsampling(Module):
    pass


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
        return sum((input - target).pow(2).sum(0))/(input.shape[1]*input.shape[2]*input.shape[3])

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

<<<<<<< HEAD
model = Sequential([Conv2d(3, 25, kernel_size = (2,2), stride = (2,2)), ReLU(), TransposeConv2d(25, 3, kernel_size = (2,2), stride = (2,2)), Sigmoid()])
=======
"""model = Sequential([Conv2d(3, 25), ReLU(), TransposeConv2d(25, 3), Sigmoid()])
>>>>>>> parent of 985abbb (sync)
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
optimizer.step()"""