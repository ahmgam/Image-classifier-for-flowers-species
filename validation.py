

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models


def validate(model, val_loader, criterion,device):
    val_loss = 0
    accuracy = 0
    if device=='cuda':
       devicee=torch.device("cuda:0")
       model.to(devicee)
    
    model.eval()
    for images, labels in val_loader:

        
        if device=='cuda':
           images, labels = images.to(devicee), labels.to(devicee)
        
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    val_loss=val_loss/len(val_loader)
    accuracy=accuracy/len(val_loader)
    return val_loss, accuracy