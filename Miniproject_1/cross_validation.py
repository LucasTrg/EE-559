from distutils.fancy_getopt import wrap_text
from re import S
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from model import *
import numpy as np
import itertools
import datetime
from torch.utils.tensorboard import SummaryWriter

class K_Fold:
    def __init__(self, dataset_1, dataset_2, k=7, batch_size=64):
        self.k=k
        self.dataset_1_chunks = dataset_1.chunk(k)
        self.dataset_2_chunks = dataset_2.chunk(k)
        self.fold_count=0
        self.batch_size = batch_size

    def fold(self):
        if self.has_next():
            train_input_chunk=torch.cat([self.dataset_1_chunks[j] for j in range(self.k) if j!=self.fold_count])
            train_target_chunk=torch.cat([self.dataset_2_chunks[j] for j in range(self.k) if j!=self.fold_count])

            test_input_chunk = self.dataset_1_chunks[self.fold_count]
            test_target_chunk = self.dataset_2_chunks[self.fold_count]
            self.fold_count+=1

            train_input_chunk_loader = torch.utils.data.DataLoader(train_input_chunk/255, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)
            train_target_chunk_loader = torch.utils.data.DataLoader(train_target_chunk/255, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)


            test_input_chunk_loader = torch.utils.data.DataLoader(test_input_chunk/255, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)
            test_target_chunk_loader = torch.utils.data.DataLoader(test_target_chunk/255, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)



            return train_input_chunk_loader, train_target_chunk_loader, test_input_chunk_loader, test_target_chunk_loader

    def has_next(self):
        return self.fold_count<self.k
        

def k_fold_CV(noisy_ds_1, noisy_ds_2,test_noisy_ds, test_clean_ds, k=10, **kwargs):

    train_loss=[]
    train_SNR=[]
    fold_generator = K_Fold(noisy_ds_1, noisy_ds_2, k=k)

    while fold_generator.has_next():

        model = Model(**kwargs)
        train_input_chunk, train_target_chunk, test_input_chunk, test_target_chunk = fold_generator.fold()



        print("Train input chunk ", len(train_input_chunk))
        print("Test input chunk ", len(test_input_chunk))

        fold_train_loss, fold_train_SNR = model.train(train_input_chunk,train_target_chunk, num_epochs=kwargs.get("num_epoch", 20))
        train_loss.append(fold_train_loss)
        train_SNR.append(fold_train_SNR)

        test_SNR=0
        for input, target in zip(test_noisy_ds, test_clean_ds):
            input=input.to(model.device).float()
            target=target.to(model.device).float()
            outputs = model.predict(input)

            test_SNR+=utils.psnr(outputs, target)
            i+=1
        


    return sum(train_loss)/len(train_loss),  sum(train_SNR)/len(train_SNR), test_SNR/i



def combination_generator(**kwargs):
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(),instance))

def grid_search(noisy_ds_1, noisy_ds_2,test_noisy_ds, test_clean_ds, cross_validation=False, **kwargs):
    results={}

    writer = SummaryWriter()
    for combination in combination_generator(**kwargs):


        print(combination)
        if cross_validation:
            res=k_fold_CV(noisy_ds_1, noisy_ds_2, test_noisy_ds, test_noisy_ds, k =4, **combination)
            writer.add_hparams({k: str(v) for k, v in combination.items()},{"Loss/train":res[0], "SNR/train":res[1], "Loss/test":res[2]})

        else :
            model = Model(**combination)
            train_input_chunk_loader = torch.utils.data.DataLoader(noisy_ds_1/255, batch_size=combination.get("batch_size",32), shuffle=False, num_workers=3, pin_memory=True)
            train_target_chunk_loader = torch.utils.data.DataLoader(noisy_ds_2/255, batch_size=combination.get("batch_size",32), shuffle=False, num_workers=3, pin_memory=True)

            test_input_chunk_loader = torch.utils.data.DataLoader(test_noisy_ds/255, batch_size=combination.get("batch_size",32), shuffle=False, num_workers=3, pin_memory=True)
            test_target_chunk_loader = torch.utils.data.DataLoader(test_noisy_ds/255, batch_size=combination.get("batch_size",32), shuffle=False, num_workers=3, pin_memory=True)

            fold_train_loss, fold_train_SNR = model.train(train_input_chunk_loader,train_target_chunk_loader, num_epochs=combination.get("num_epoch", 20))
            test_SNR=model.measureSNR(test_input_chunk_loader,test_target_chunk_loader)

            writer.add_hparams({k: str(v) for k, v in combination.items()},{"Loss/train":fold_train_loss, "SNR/train":fold_train_SNR, "Loss/test":test_SNR})
            print(test_SNR)


    writer.close()
    f = open("gridsearch-{}.txt".format(datetime.datetime.now()), "x") 
    f.write(str(results))
    return results




if __name__ == "__main__":
    batch_size=32
    
    parameters = {
        "num_epoch" : [10,20], 
        "lr":[4e-3, 1e-3, 5e-4],
        "model":[UNet],#MiniEncoder
        "loss":[nn.MSELoss, nn.L1Loss,nn.HuberLoss],
        "batch_size":[32,64],
        "channel_number":[96, 192]
    }
   
    device = torch.device ( "cuda" if torch.cuda.is_available() else "cpu" )

    train_input_dataset , train_target_dataset = torch.load ("train_data.pkl")
    test_input_dataset, test_target_dataset = torch.load("val_data.pkl")
    

    print(grid_search(train_input_dataset, train_target_dataset,test_input_dataset, test_target_dataset,**parameters))
    #print(k_fold_CV(train_input_dataset,train_target_dataset,k=5, kwargs=kwargs))
    
