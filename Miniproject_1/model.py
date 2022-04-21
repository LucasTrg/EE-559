from pickletools import optimize
from xml.etree.ElementTree import C14NWriterTarget
import torch
from torch import nn
import utils


class Model():
    def __init__(self, **kwargs) -> None:

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

        self.criterion = nn.MSELoss()
        ##Define all layers here 

        input_channel_number = 3

        self.enc_conv0 = nn.Conv2d(in_channels=input_channel_number, out_channels=48, kernel_size=3)

        self.enc_conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)

        self.upsample5 = nn.Upsample(size=2, mode="nearest")
        
        self.dec_conv5A = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3)
        self.dec_conv5B = nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3)

        self.upsample4 = nn.Upsample(size=2, mode="nearest")

        self.dec_conv4A = nn.Conv2d(in_channels=144, out_channels=48, kernel_size=3)
        self.dec_conv4B = nn.Conv2d(in_channels=144, out_channels=48, kernel_size=3)

        self.upsample3 = nn.Upsample(size=2, mode="nearest")

        self.dec_conv4A = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3)
        self.dec_conv4B = nn.Conv2d(in_channels=144, out_channels=48, kernel_size=3)


        #Yada yada yada

    def forward(self, features):
        activation = self.conv1(features)
        activation = torch.relu(activation)
        #Yayaya
        return activation


    def load_pretrained_model(self)-> None:
        pass

    def train(self, train_input, train_target, num_epochs)-> None:

        for epoch in range(num_epochs):
            loss = 0 
            for batch_features, batch_target in zip(train_input, train_input):
                batch_features = batch_features.view(-1, 784).to(self.device)
                self.optimizer.zero_grad()

                outputs = model(batch_features)

                train_loss = self.criterion(outputs, batch_target)

                train_loss.backward()

                self.optimizer.step()

                loss += train_loss.item()

            loss = loss /len(self.training_loader)
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        

    def predict(self, test_input)-> torch.Tensor:
        return self(test_input)

if __name__ == "__main__":
    device = torch.device ( "cuda" if torch.cuda.is_available() else "cpu" )
    
    
    train_input_dataset , train_target_dataset = torch.load ("train_data.pkl")
    train_input_loader = torch.utils.data.DataLoader(
    train_input_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
    train_target_loader = torch.utils.data.DataLoader(
    train_target_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

    model = Model(1).to(device)
    model.train()