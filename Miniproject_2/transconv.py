from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import modules


class TransposeConv2d(modules.Module):
    ### Transposed convolutional layer, works in the same manner as torch.nn.ConvTranspose2d : https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    def __init__(self, in_channels, out_channels, kernel_size = (3,3), stride = (1,1), dilation=(1,1), padding = (0,0), bias = True):
        ## instantiate parameters
        super().__init__()
        self.kernel_size = kernel_size; self.bias = bias; self.dilation=dilation; self.padding = padding; self.stride = stride
        self.in_channels = in_channels; self.out_channels = out_channels

        # compute the "real" padding applied to the input using the padding and kernel size
        self.real_padding = (dilation[0]*(kernel_size[0] - 1) - padding[0], dilation[1]*(kernel_size[1] - 1) - padding[1])

        # compute k for the uniform distribution
        k = 1/(out_channels*kernel_size[0]*kernel_size[1])

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
        #input_reshape = self.input.reshape(self.input.shape[0],self.out_channels,self.input.shape[2]).transpose(1,2)
        #maybe reshape there

        # insert zeros between the columns and rows the input according to the stride
        zeroInsertDim2 = self.stride[0]*input.shape[2] if(self.stride[0]>1)  else input.shape[2]
        zeroInsertDim3 = self.stride[1]*input.shape[3] if(self.stride[1]>1)  else input.shape[3]
        zeroInsertDim1 = input.shape[1]//self.stride[0]#if we say that the stride is even for 0 and 1 dim

        inputZeroInserted = empty(input.shape[0], zeroInsertDim1, zeroInsertDim2, zeroInsertDim3).zero_()
        inputZeroInserted[:,:,::self.stride[0], ::self.stride[1]] = self.input[:,::self.stride[0],:,:]

        # the convolution is performed as a matrix multiplication after unfolding the input and reshaping the weights
        #self.unfolded = unfold(self.input, kernel_size = self.kernel_size, dilation=self.dilation, padding = self.real_padding)

        # compute the output dimensions
        outDim2 = (input.shape[2]-1)*self.stride[0] - 2*self.real_padding[0] + self.dilation[0]*(self.kernel_size[0] - 1) + 1  
        outDim3 = (input.shape[3]-1)*self.stride[1]  - 2*self.real_padding[1] + self.dilation[1]*(self.kernel_size[1] - 1) + 1

        self.folded=fold(self.input, (self.input.shape[0], self.out_channels, outDim2, outDim3), (self.kernel_size[0], self.kernel_size[1]), dilation= self.dilation, padding=self.real_padding, stride = self.stride)
        print('zero inserted:', inputZeroInserted.shape, self.input.shape, self.folded.shape, 'wtranspose: ', self.w.view(self.out_channels, -1).t().shape)
        out = self.w.view(-1, self.out_channels).t() @ inputZeroInserted

        # if bias = true, add the bias after reshaping accordingly
        if(self.bias):
            out.add(self.b.view(1,-1,1))

      

        

        # return the output after reshaping in the correct dimensions
        return out
            #input.shape[0], self.out_channels, outDim2 , outDim3)

    def backward(self, gradwrtoutput):
        ## backward pass of the module
        # the gradient of the output is reshaped to be able to perform a convolution using matrix multiplication
        grad_reshape = gradwrtoutput#.reshape(gradwrtoutput.shape[0],self.out_channels,self.unfolded.shape[2]).transpose(1,2)

        # the gradient of the weight is computed through a convolution between the output gradient and the input
        # the result is summed and reshaped according to the weight dimensions
        self.dw = (grad_reshape @ self.folded).sum(0).view(self.dw.shape)

        # if bias = true, get the gradient of the bias by summing the gradient of the output along the first, third and fourth dimension
        if(self.bias):
            self.db = gradwrtoutput.sum((0,2,3))

        # return the gradient of the input by performing a convolution between the weights and the gradient of the ouput
        # the result is reshaped and folded to return the correct dimension
        gw = (grad_reshape@self.w.view(self.out_channels, -1)).transpose(1,2)
        print(gw.shape, self.input.shape[2], self.input.shape[3])
        return unfold(gw, (self.input.shape[2], self.input.shape[3]), (self.kernel_size[0], self.kernel_size[1]), dilation=self.dilation, padding=self.real_padding)

    def param(self):
        ## return the parameters values and gradients by pairs
        if(self.bias):
            return [(self.w, self.dw), (self.b, self.db)]
        else:
            return [(self.w, self.dw)]
