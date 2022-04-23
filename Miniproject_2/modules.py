from turtle import forward
import torch
import math

torch.set_grad_enabled(False)



class Module(object):
    def forward (self,*input):
        raise NotImplementedError

    def backward (self,*gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []



class Convolution(Module):
    def __init__(self, in_channels, out_channels, kernel_size = (2,2), bias = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        k = 1/(in_channels*kernel_size[0]*kernel_size[1])
        self.w = torch.FloatTensor(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(-math.sqrt(k), math.sqrt(k))
        self.dw = torch.zeros_like(self.w)
        if(bias):
            self.b = torch.FloatTensor(out_channels).uniform_(-math.sqrt(k), math.sqrt(k))
        else:
            self.b = torch.zeros(out_channels)

        self.db = torch.zeros_like(self.b)
 
    def forward(self, input):
        self.input = input
        self.unfolded = torch.nn.functional.unfold(self.input, kernel_size = self.kernel_size)
        wxb = self.w.view(self.out_channels, -1) @ self.unfolded + self.b.view(1,-1,1)
        return wxb.view(input.shape[0], self.out_channels, input.shape[2] - self.kernel_size[0] + 1 , input.shape[3] - self.kernel_size[1]+ 1)

    def backward(self, gradwrtoutput):
        self.dw.add(self.unfolded.t() @ gradwrtoutput)
        if(self.bias):
            self.db.add(gradwrtoutput.sum(0))
        return gradwrtoutput @ self.w.view(self.out_channels,-1).t()

    def param(self):
        return [(self.w, self.dw), (self.b, self.db)]

class ReLu(Module):
    def forward (self, input):
        self.input = input
        return torch.max(torch.zeros_like(input),input)

    def backward(self, gradwrtoutput):
        dsigma = torch.zeros_like(self.input)
        dsigma[self.input > 0] = 1
        return dsigma * gradwrtoutput

class Sigmoid(Module):
    def forward (self,input):
        self.input = input
        return 1/(1+torch.exp(-input))

    def backward(self, gradwrtoutput):
        sigma = 1/(1+torch.exp(-input))
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
        return (input - target).pow(2).mean

    def backward(self):
        return 2*(self.input-self.target)/self.input.size(0)


class SGDOptimizer():
    def __init__(self, param, eta = 0.01):
        self.param = param
        self.eta = eta

    def step(self):
        for param in self.param:
            weight, grad = param
            weight.add(-self.eta*grad)

model = Sequential([Convolution(2, 25), ReLu(), Convolution(25, 1), Sigmoid()])

n = 1000
center = 0.5
radius = 1 / math.sqrt((2 * math.pi))

random_tensor = torch.empty((100, 2, 32, 32)).uniform_(0, 1)
radius_sq = math.pow(radius, 2)

temp_tensor = random_tensor.sub(center).pow(2).sum(1)
target_tensor = torch.where(temp_tensor < radius_sq, 1, 0)

log_losses = []
mean_losses = 0

output = model.forward(random_tensor)
criterion = MSELoss()
            
loss = criterion.forward(output, target_tensor)
mean_losses += loss.mean().item()

#model.zero_grad()

loss_grad = MSELoss.backward()
model.backward(loss_grad)

SGDOptimizer.step()