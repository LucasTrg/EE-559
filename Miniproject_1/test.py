import torch
import model
if __name__=="__main__":
    Device = torch.device ( "cuda" if torch.cuda.is_available() else "cpu" ) 
    model = model.Model()
    model.load_pretrained_model("V1.pt")
    batch_size=128
    test_input_dataset , test_target_dataset = torch.load ("val_data.pkl")
    test_input_loader = torch.utils.data.DataLoader(
        test_input_dataset/256, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_target_loader = torch.utils.data.DataLoader(
        test_target_dataset/256, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

