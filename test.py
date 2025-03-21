import torch

model_data = torch.load("DenseNet121_64_lr0.0003_0-5.pt", map_location=torch.device('cpu'))

if isinstance(model_data, torch.nn.Module):
    print("You saved the full model.")
elif isinstance(model_data, dict):
    print("You saved a state_dict.")
else:
    print("Unknown format.")
