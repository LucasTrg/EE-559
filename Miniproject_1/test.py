import torch
import model
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import utils 

if __name__=="__main__":
    matplotlib.use('TkAgg')
    Device = torch.device ( "cuda" if torch.cuda.is_available() else "cpu" ) 
    model = model.Model()
    model.load_pretrained_model("V3.pt")
    batch_size=128
    test_input_dataset , test_target_dataset = torch.load ("val_data.pkl")
    test_input_loader = torch.utils.data.DataLoader(
        test_input_dataset/255, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    test_target_loader = torch.utils.data.DataLoader(
        test_target_dataset/255, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)



    SNR=0
    i=0
    for input, target in zip(test_input_loader, test_target_loader):
        input=input.to(Device).float()
        target=target.to(Device).float()

        #for (img1,img2) in zip (input, target):

        #    SNR+=utils.psnr(img1, img2)
        #    i+=1
        plt.subplot(131)
        plt.imshow(  input[0].cpu().detach().permute(1, 2, 0)  )
        plt.subplot(132)
        print(model.predict(input).shape)
        print( model.predict(input)[0])
        plt.imshow(model.predict(input)[0].cpu().detach().permute(1, 2, 0))
        plt.subplot(133)
        plt.imshow(  target[0].cpu().detach().permute(1, 2, 0)  )
        plt.show()
    #print("Average SNR :", SNR/i)



    #print(model.predict(test_input_dataset.to(Device).float()).cpu().detach()[0])