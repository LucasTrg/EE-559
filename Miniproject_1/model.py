from pickletools import optimize
from xml.etree.ElementTree import C14NWriterTarget
import torch
from torch import nn
import utils


class VeryMiniEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        input_channel_number = 3
        self.enc_conv0 = nn.Conv2d(in_channels=input_channel_number, out_channels=16, kernel_size=3, padding="same", stride=1)
        self.relu0 = nn.LeakyReLU(negative_slope=0.01)
        self.enc_conv1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding="same",stride=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, features):
        encoding=self.relu0(self.enc_conv0(features))
        return self.sigmoid(self.enc_conv1(encoding))


class MiniEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        input_channel_number = 3
        self.enc_conv0 = nn.Conv2d(in_channels=input_channel_number, out_channels=48, kernel_size=3, padding="same", stride=1)
        self.relu0 = nn.LeakyReLU(negative_slope=0.1)
        
        self.enc_conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same",stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding="same")
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)


        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv1A = nn.Conv2d(in_channels=48+input_channel_number, out_channels=64, kernel_size=3, padding="same")
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

        encoding = self.relu2(self.enc_conv2(encoding))
        #print("After conv 2", encoding.shape)

        decoding =  self.upsample1(encoding)
        #print("After upsample 1", decoding.shape)

        decoding=torch.cat((decoding, stack.pop()), axis=1)
        #print("After cat 1", decoding.shape)

        decoding=self.relu1A(self.dec_conv1A(decoding))        
        decoding=self.relu1B(self.dec_conv1B(decoding)) 
        decoding=self.dec_conv1C(decoding)

        return decoding


class UNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        ##Define all layers here 

        input_channel_number = 3
        channel_number = kwargs.get("channel_number",96)
        self.batch_norm=kwargs.get("batch_norm", True)
        self.enc_conv0 = nn.Conv2d(in_channels=input_channel_number, out_channels=int(channel_number/2), kernel_size=3, padding="same", stride=1)
        self.relu0 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv1 = nn.Conv2d(in_channels=int(channel_number/2), out_channels=int(channel_number/2), kernel_size=3, padding="same",stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv2 = nn.Conv2d(in_channels=int(channel_number/2), out_channels=int(channel_number/2), kernel_size=3, padding="same", stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv3 = nn.Conv2d(in_channels=int(channel_number/2), out_channels=int(channel_number/2), kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3= nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv4 = nn.Conv2d(in_channels=int(channel_number/2), out_channels=int(channel_number/2), kernel_size=3, padding="same")
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv5 = nn.Conv2d(in_channels=int(channel_number/2), out_channels=int(channel_number/2), kernel_size=3, padding="same")
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.1)

        self.enc_conv6 = nn.Conv2d(in_channels=int(channel_number/2), out_channels=int(channel_number/2), kernel_size=3, padding="same")
        self.relu6 = nn.LeakyReLU(negative_slope=0.1)

        self.upsample5 = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.dec_conv5A = nn.Conv2d(in_channels=int(channel_number), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.BN5 = nn.BatchNorm2d(channel_number)
        self.relu5A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv5B = nn.Conv2d(in_channels=int(channel_number), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.relu5B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample4 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv4A = nn.Conv2d(in_channels=int(channel_number*3/2), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.BN4 = nn.BatchNorm2d(channel_number)
        self.relu4A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv4B = nn.Conv2d(in_channels=int(channel_number), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.relu4B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv3A = nn.Conv2d(in_channels=int(channel_number*3/2), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.BN3 = nn.BatchNorm2d(channel_number)
        self.relu3A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv3B = nn.Conv2d(in_channels=int(channel_number), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.relu3B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv2A = nn.Conv2d(in_channels=int(channel_number*3/2), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.BN2 = nn.BatchNorm2d(channel_number)
        self.relu2A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv2B = nn.Conv2d(in_channels=int(channel_number), out_channels=int(channel_number), kernel_size=3, padding="same")
        self.relu2B = nn.LeakyReLU(negative_slope=0.1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

        self.dec_conv1A = nn.Conv2d(in_channels=channel_number+input_channel_number, out_channels=64, kernel_size=3, padding="same")
        self.BN1 = nn.BatchNorm2d(64)
        self.relu1A = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv1B = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.relu1B = nn.LeakyReLU(negative_slope=0.1)
        self.dec_conv1C = nn.Conv2d(in_channels=32, out_channels=input_channel_number, kernel_size=3, padding="same")
        self.linear_act = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

        

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


        #print("DECODING SHAPE ", decoding.shape)
        #print("STACK POP SHAPE", stack[-1].shape)
        decoding=torch.cat((decoding, stack.pop()), axis=1)

        if self.batch_norm:
            decoding=self.relu5A(self.BN5(self.dec_conv5A(decoding)))
        else:
            decoding=self.relu5A(self.dec_conv5A(decoding))

        decoding=self.relu5B(self.dec_conv5B(decoding))        
        #print("After dec 5", decoding.shape)
        
        decoding = self.upsample4(decoding) 
        #print("After up 4", decoding.shape)

        decoding=torch.cat((decoding, stack.pop()), axis=1)
        if self.batch_norm:
            decoding=self.relu4A(self.BN4(self.dec_conv4A(decoding)))        
        else:
            decoding=self.relu4A(self.dec_conv4A(decoding))

        decoding=self.relu4B(self.dec_conv4B(decoding))   
        #print("After dec 4", decoding.shape)


        decoding = self.upsample3(decoding) 
        decoding=torch.cat((decoding, stack.pop()), axis=1)
        if self.batch_norm:
            decoding=self.relu3A(self.BN3(self.dec_conv3A(decoding)))        
        else:
            decoding=self.relu3A(self.dec_conv3A(decoding))     

        decoding=self.relu3B(self.dec_conv3B(decoding)) 
        #print("After dec 3", decoding.shape)

        decoding = self.upsample2(decoding) 
        decoding=torch.cat((decoding, stack.pop()), axis=1)

        if self.batch_norm:
            decoding=self.relu2A(self.BN2(self.dec_conv2A(decoding)))        
        else:
            decoding=self.relu2A(self.dec_conv2A(decoding))       

        decoding=self.relu2B(self.dec_conv2B(decoding))    
        #print("After dec 2", decoding.shape)

        decoding = self.upsample1(decoding)
        decoding=torch.cat((decoding, stack.pop()), axis=1)


        if self.batch_norm:
            decoding=self.relu1A(self.BN1(self.dec_conv1A(decoding)))        
        else:
            decoding=self.relu1A(self.dec_conv1A(decoding))     

        decoding=self.relu1B(self.dec_conv1B(decoding)) 
        decoding=self.dec_conv1C(decoding)
        decoding=self.linear_act(decoding)

        return decoding




class Model():
    def __init__(self,TBWriter=None, foldCount=None,**kwargs) -> None:


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.autoencoder = kwargs.get("model",UNet)(**kwargs).to(self.device)
        self.criterion = kwargs.get("loss", nn.MSELoss)()
        self.optimizer=torch.optim.Adam(self.autoencoder.parameters(),lr=kwargs.get("lr", 1e-3))
        self.TBWriter = TBWriter
        self.foldCount=None
        ##Define all layers here 


        

    

    def load_pretrained_model(self, path,**kwargs)-> None:
        model = UNet(*kwargs)
        model.load_state_dict(torch.load(path))
        model.eval().to(self.device)
        self.autoencoder=model

    def train(self, train_input, train_target, num_epochs)-> None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=11)

        

        for epoch in range(num_epochs):
            loss = 0 
            SNR = 0
            for batch_features, batch_target in zip(train_input, train_target):
                #print(batch_features.shape)
                batch_features = batch_features.float().to(self.device)
                batch_target = batch_target.float().to(self.device)

                self.optimizer.zero_grad()

                outputs = self.autoencoder(batch_features)

                train_loss = self.criterion(outputs, batch_target)

                train_loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                loss += train_loss.item()
                SNR += utils.psnr(outputs, batch_target)
            loss = loss /len(train_target)
            SNR = SNR / len(train_target)
            if self.TBWriter is not None:
                self.TBWriter("Loss/train", loss, num_epochs)
                self.TBWriter("SNR/train", SNR, num_epochs)
            
            #self.TBWriter.add_scalars('runs_split   {}'.format(self.foldCount) if self.foldCount else "run", {'Loss/train': training_loss,
            #                                'Loss/validation': validation_loss}, epoch+1)
            print("epoch : {}/{}, loss = {:.6f}, SNR = {:.3f}, lr={}".format(epoch + 1, num_epochs, loss, SNR, self.optimizer.param_groups[0]['lr']))
            if (epoch+1)%10==0:
                print("Saving checkpoint for the model")
                torch.save(self.autoencoder.state_dict(), "V5-big.pt")
        return loss, SNR


    def measureSNR(self, test_input, test_target):
        SNR=0
        torch.cuda.empty_cache() 
        with torch.no_grad():
            for batch_test_input, batch_test_target in zip(test_input,test_target):
                batch_test_input = batch_test_input.float().to(self.device)
                output=self.autoencoder(batch_test_input)
                del batch_test_input
                batch_test_target = batch_test_target.float().to(self.device)
                SNR+=utils.psnr(output, batch_test_target)
            return SNR/len(test_input)

    ### Add support to dataloader
    def predict(self, test_input)-> torch.Tensor:
        return self.autoencoder(test_input) 




if __name__ == "__main__":
    batch_size=32

    device = torch.device ( "cuda" if torch.cuda.is_available() else "cpu" )
    
    print("PyTorch version : ",torch.__version__)
    train_input_dataset , train_target_dataset = torch.load ("train_data.pkl")
    train_input_loader = torch.utils.data.DataLoader(
        train_input_dataset/255, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_target_loader = torch.utils.data.DataLoader(
        train_target_dataset/255, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = Model()
    model.train(train_input_loader, train_target_loader, 100)
