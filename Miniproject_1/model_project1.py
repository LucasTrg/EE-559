import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import utils

class AutoEncoder(nn.Module):

  def __init__(self, in_channels=3, out_channels=3, skip_connections=False, batch_normalization=False):
    super().__init__()

    self.conv1=nn.Conv2d(in_channels, 150,
                               kernel_size = 5,
                               padding = 'same')
    
    self.pool=nn.MaxPool2d(kernel_size=2)
    
    self.bn1 = nn.BatchNorm2d(150)

    
    self.conv2=nn.Conv2d(150, 100,
                               kernel_size = 4,
                               padding = 'same')#'same'
    
    self.bn2 = nn.BatchNorm2d(100)
    
    self.conv3=nn.Conv2d(100, 80,
                               kernel_size = 3,
                               padding = 'same' )
    self.bn3 = nn.BatchNorm2d(80)
    
    self.conv4=nn.Conv2d(80, 50,
                               kernel_size = 2,
                               padding = 'same')
    
    self.deconv1=nn.Conv2d(50,80,
                                    kernel_size=2,
                                    padding='same')
    self.deconvbn1=nn.BatchNorm2d(80)
    
    self.deconv2=nn.Conv2d(80,100,
                                    kernel_size=3,
                                    padding='same')
    
    self.deconvbn2 = nn.BatchNorm2d(100)
    
    self.deconv3=nn.Conv2d(100,150,
                                    kernel_size=4,
                                    padding='same')
    
    self.deconvbn3 = nn.BatchNorm2d(150)

    
    self.deconv4=nn.Conv2d(150,out_channels,
                                    kernel_size=5,
                                    padding='same')
    
    self.upsampling=nn.UpsamplingNearest2d(scale_factor=2)
    

    
    self.skip_connections=skip_connections
    self.batch_normalization=batch_normalization



  def forward(self,x):

    y=self.conv1(x)
    y=self.pool(y)
    y=F.leaky_relu(y)
    if self.batch_normalization: y =F.leaky_relu(self.bn1(y))
    y=F.leaky_relu(self.conv2(y))
    if self.batch_normalization: y =F.leaky_relu(self.bn2(y))
    y=F.leaky_relu(self.conv3(y))
    if self.batch_normalization: y =F.leaky_relu(self.bn3(y))
    y=F.leaky_relu(self.conv4(y))

    y=F.leaky_relu(self.deconv1(y))
    if self.batch_normalization: y =F.leaky_relu(self.deconvbn1(y))
    y=F.leaky_relu(self.deconv2(y))
    if self.batch_normalization: y =F.leaky_relu(self.deconvbn2(y))
    y=F.leaky_relu(self.deconv3(y))
    if self.batch_normalization: y =F.leaky_relu(self.deconvbn3(y))
    y=F.leaky_relu(self.upsampling(y))
    y=self.deconv4(y)
    if self.skip_connections:
      #print(y.size(), F.leaky_relu(self.pool(self.conv1(x))).size())
      y =y + x
    y=torch.sigmoid(y)

    return y

class Model():
    def __init__(self, **kwargs) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = AutoEncoder(**kwargs).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.autoencoder.parameters(),lr=kwargs.get("lr", 1e-3))

    def load_pretrained_model(self, path,**kwargs)-> None:
        model = AutoEncoder(*kwargs)
        model.load_state_dict(torch.load(path))
        model.eval().to(self.device)
        self.autoencoder=model

    def train(self, train_input, train_target, num_epochs) -> None:
        self.autoencoder.train()
        criterion=nn.MSELoss()
        optimizer=optim.Adamax(self.autoencoder.parameters(), lr = 1e-3)
        mini_batch_size=200
        train_loss=0
        SNR=0


        for e in range(num_epochs):
             for b in range(0, train_input.size(0),mini_batch_size):
                 output=self.autoencoder(train_input.narrow(0, b, mini_batch_size))
                 target=train_target.narrow(0, b, mini_batch_size)

                 loss=criterion(output, target)

                 self.autoencoder.zero_grad()
                 loss.backward()
                 optimizer.step()

                 train_loss+=loss.item()
                 SNR+=utils.psnr(output, target).item()
             train_loss=train_loss/len(train_target)
             SNR=SNR/(train_input.size(0)//mini_batch_size)
        
        print('epoch : ',e, ', loss = ',train_loss)

    def predict(self, test_input)-> torch.Tensor:
        return self.autoencoder(test_input) 