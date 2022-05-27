from torch import set_grad_enabled, load
import pickle 
import modules
import math

set_grad_enabled(False)


### For mini - project 2
class Model () :

    def __init__(self) -> None:
     ## instantiate model + optimizer + loss function + any other stuff you need

     #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     self.model=modules.Sequential([modules.Conv2d(3, 10,kernel_size=(2,2), stride=(2,2)), modules.ReLU(), modules.Conv2d(10, 20, kernel_size=(2,2), stride=(2,2)),
                                    modules.ReLU(), modules.TransposeConv2d(20, 10,kernel_size=(2,2), stride=(2,2)), modules.ReLU(),
                                    modules.TransposeConv2d(10, 3, kernel_size=(2,2), stride=(2,2)), modules.Sigmoid()])
     self.criterion=modules.MSE()
     self.optimizer=modules.SGD()
    



    def load_pretrained_model (self, path) -> None :
     ## This loads the parameters saved in bestmodel .pth into the model
     model=Model()
     #model.load(load(path))
     self.model=model
     with open(path, 'rb') as f:
         parameters=pickle.load(f)
         print(len(parameters))
     self.model.set_param(parameters)
     

    
        
     
     

    def train(self , train_input , train_target , num_epochs ) -> None :
     #: train input : tensor of size (N, C, H, W) containing a noisy version of the images
     #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the
     # same images , which only differs from the input by their noise 
     for epoch in range(num_epochs):

         train_loss=0
         SNR=0
         batch_size=200

         for batch in range(0, train_input.shape[0], batch_size):
             output=self.model.forward(train_input.narrow(0, batch, batch_size))

             #print('param after output :',self.model.param())
             loss=self.criterion.forward(output, train_target.narrow(0, batch, batch_size))

             loss_grad=self.criterion.backward()
             self.model.backward(loss_grad)
             self.optimizer.step(self.model.param())
             #print(loss.item())
             train_loss+=loss.item()
             SNR+=self.PSNR(output, train_target.narrow(0, batch, batch_size))
        
         train_loss=train_loss/len(train_target)
         SNR=SNR/(train_input.size(0)//batch_size)
        
         print('epoch : ',epoch, ', loss = ',train_loss , 'PSNR = ', SNR)
         if (epoch)%10==0:
                print("Saving checkpoint for the model")
                #for param in:
                with open('try_again.pth', 'wb') as f:
                 pickle.dump( self.model.param(), f)

    def predict(self , test_input ):
     #: test˙input : tensor of size (N1 , C, H, W) with values in range 0 -255 that has to
     # be denoised by the trained or the loaded network .
     # #: returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.
     return self.model.forward(test_input)

    def PSNR(self,noisy_im, clean_im):
     # Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
     mse = self.criterion.forward(noisy_im, clean_im)
     return -10 * math.log10(mse + 10** -8)


if __name__=="__main__":

    train_input_data , train_target_data = load ("train_data.pkl")
    test_input_data , test_target_data = load ("val_data.pkl")
    #device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')


    train_input=train_input_data/255
    train_target=train_target_data/255
    test_input=test_input_data/255
    test_target=test_target_data/255
    


    model=Model()
    model.train(train_input.float(), train_target.float(), 2)
    test_output=model.predict(test_input.float())
    PSNR=model.PSNR(test_output, test_target)
    model.load_pretrained_model('try_again.pth')
    print('PSNR test set = ', PSNR)
    



    

   