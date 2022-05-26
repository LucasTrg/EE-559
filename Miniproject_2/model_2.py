import modules
import torch

torch.set_grad_enabled(False)

class Model():
    def __init__(self) -> None:
        super().__init__()

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model=modules.Sequential([modules.Conv2d(3, 25,kernel_size=(2,2),stride=(2,2)),
                            modules.ReLU(),
                            modules.Conv2d(25, 60, kernel_size=(2,2),stride=(2,2)),
                            modules.ReLU(),
                            modules.TransposeConv2d(60,25, kernel_size=(2,2),stride=(2,2)),
                            modules.ReLU(),
                            modules.TransposeConv2d(25,3, kernel_size=(2,2),stride=(2,2)),
                            modules.Sigmoid()
                            ])
        self.criterion=modules.MSE()
        self.optimizer=modules.SGD(self.model.param())

        

    def load_pretrained_model(self) -> None:
        pass

    def train(self, train_input, train_target) -> None:
        epochs=30
        batch_size=200

        for epoch in range(epochs):
            loss=0

            for batch in range(0,train_input.size(0), batch_size):

                output=self.model.forward(train_input.narrow(0, batch, batch_size))
                target=train_target.narrow(0, batch, batch_size)
                #print(output.size(), target.size())
                train_loss=self.criterion.forward(output, target)

                loss_grad=self.criterion.backward()
                #print(loss_grad)
                self.model.backward(loss_grad)
                self.optimizer.step()
                loss+=train_loss.item()
            
            loss=loss/len(train_target)




    

    def predict(self, test_input) -> torch.Tensor:
        return self.model(test_input)



    
    def measurePSNR(self, test_input, test_target):
         psnr=0
         batch_size=200

         for b in range(0, test_input.size(0), batch_size):
             denoised=self.model(test_input.narrow(0, b, batch_size))
             for i in range(batch_size): 
                 psnr+=PSNR(denoised[i], test_target[b+i])
         return psnr/len(test_target)


def PSNR(noisy_im, clean_im):
        mse=modules.MSE(noisy_im,clean_im)
        return -10*torch.log10(mse+10**-8)
    

if __name__=="__main__":
    batch_size=200
    train_input_dataset , train_target_dataset = torch.load ("train_data.pkl")
    train_input_loader=train_input_dataset/255
    train_target_loader=train_target_dataset/255


    model=Model()
    model.train(train_input_loader, train_target_loader)






