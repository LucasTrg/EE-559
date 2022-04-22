from pickletools import optimize
from xml.etree.ElementTree import C14NWriterTarget
import torch
from torch import nn
import utils


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        ##Define all layers here 

        input_channel_number = 3

        self.enc_conv0 = nn.Conv2d(in_channels=input_channel_number, out_channels=48, kernel_size=3, padding="same", stride=1)
        self.relu0 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same",stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same", stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3= nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)

        self.upsample5 = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.dec_conv5A = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding="same")
        self.relu5A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv5B = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding="same")
        self.relu5B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample4 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv4A = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding="same")
        self.relu4A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv4B = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding="same")
        self.relu4B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv3A = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding="same")
        self.relu3A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv3B = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding="same")
        self.relu3B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv2A = nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding="same")
        self.relu2A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv2B = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding="same")
        self.relu2B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv1A = nn.Conv2d(in_channels=96+input_channel_number, out_channels=64, kernel_size=3, padding="same")
        self.relu1A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv1B = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.relu1B = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv1C = nn.Conv2d(in_channels=32, out_channels=input_channel_number, kernel_size=3, padding="same")

    def forward(self, features):
        stack = [features]
        #print("features", features.shape)

        ###ENCODING###
        encoding = self.relu0(self.enc_conv0(features))
        encoding = self.relu1(self.enc_conv1(encoding))
        encoding = self.pool1(encoding)
        #print("After conv 1", encoding.shape)
        stack.append(encoding)

        encoding = self.pool2(self.relu2(self.enc_conv2(encoding)))
        #print("After conv 2", encoding.shape)

        stack.append(encoding)

        encoding = self.pool3(self.relu3(self.enc_conv3(encoding)))
        #print("After conv 3", encoding.shape)

        stack.append(encoding)

        encoding = self.pool4(self.relu4(self.enc_conv4(encoding)))
        #print("After conv 4", encoding.shape)

        stack.append(encoding)

        encoding = self.pool5(self.relu5(self.enc_conv5(encoding)))

        encoding = self.relu6(self.enc_conv6(encoding))

        ####DECODING###
        decoding = self.upsample5(encoding)

        decoding=torch.cat((decoding, stack.pop()), axis=1)
        decoding=self.relu5A(self.dec_conv5A(decoding))        
        decoding=self.relu5B(self.dec_conv5B(decoding))        
        #print("After dec 5", decoding.shape)
        
        decoding = self.upsample4(decoding) 
        #print("After up 4", decoding.shape)

        decoding=torch.cat((decoding, stack.pop()), axis=1)
        decoding=self.relu4A(self.dec_conv4A(decoding))        
        decoding=self.relu4B(self.dec_conv4B(decoding))   
        #print("After dec 4", decoding.shape)


        decoding = self.upsample3(decoding) 
        decoding=torch.cat((decoding, stack.pop()), axis=1)
        decoding=self.relu3A(self.dec_conv3A(decoding))        
        decoding=self.relu3B(self.dec_conv3B(decoding)) 
        #print("After dec 3", decoding.shape)

        decoding = self.upsample2(decoding) 
        decoding=torch.cat((decoding, stack.pop()), axis=1)
        decoding=self.relu2A(self.dec_conv2A(decoding))        
        decoding=self.relu2B(self.dec_conv2B(decoding))    
        #print("After dec 2", decoding.shape)

        decoding = self.upsample1(decoding)
        decoding=torch.cat((decoding, stack.pop()), axis=1)
        decoding=self.relu1A(self.dec_conv1A(decoding))        
        decoding=self.relu1B(self.dec_conv1B(decoding)) 
        decoding=self.dec_conv1C(decoding)

        return decoding




class Model():
    def __init__(self, **kwargs) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = AutoEncoder(*kwargs).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.autoencoder.parameters(),lr=1e-3)
        
        ##Define all layers here 


        

    

    def load_pretrained_model(self)-> None:
        pass

    def train(self, train_input, train_target, num_epochs)-> None:

        for epoch in range(num_epochs):
            loss = 0 
            for batch_features, batch_target in zip(train_input, train_target):
                #print(batch_features.shape)
                batch_features = batch_features.to(self.device).float()
                batch_target = batch_target.to(self.device).float()

                self.optimizer.zero_grad()

                outputs = self.autoencoder(batch_features)

                train_loss = self.criterion(outputs, batch_target)

                train_loss.backward()

                self.optimizer.step()

                loss += train_loss.item()

            loss = loss /len(train_target)
            print("epoch : {}/{}, loss = {:.6f}, SNR = {:.3f}".format(epoch + 1, num_epochs, loss, utils.psnr(outputs, batch_target)))
        

    def predict(self, test_input)-> torch.Tensor:
        return AutoEncoder(test_input)

if __name__ == "__main__":
    device = torch.device ( "cuda" if torch.cuda.is_available() else "cpu" )
    
    print("PyTorch version : ",torch.__version__)
    train_input_dataset , train_target_dataset = torch.load ("train_data.pkl")
    train_input_loader = torch.utils.data.DataLoader(
    train_input_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    train_target_loader = torch.utils.data.DataLoader(
    train_target_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = Model()
    model.train(train_input_loader, train_target_loader, 50)