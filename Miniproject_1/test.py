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
    model.load_pretrained_model("V5-big.pt")
    batch_size=128
    test_input_dataset , test_target_dataset = torch.load ("val_data.pkl")
    test_input_loader = torch.utils.data.DataLoader(
        test_input_dataset/255, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    test_target_loader = torch.utils.data.DataLoader(
        test_target_dataset/255, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)



    SNR=0
    cSNR=0
    SNRin=0


    i=0
    for input, target in zip(test_input_loader, test_target_loader):
        input=input.to(Device).float()
        target=target.to(Device).float()
        outputs = model.predict(input)

        cSNR+=utils.psnr(outputs, target)
        SNRin+=utils.psnr(input, target)
        i+=1
           #cSNR+=utils.psnr(model.predict(img1)[0], img2)



    
    print("Average original SNR :", SNRin)
    print("Average compensated SNR :", cSNR/i)




    #print(model.predict(test_input_dataset.to(Device).float()).cpu().detach()[0])