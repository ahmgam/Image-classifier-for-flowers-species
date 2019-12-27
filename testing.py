

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models

def test(model, testloader, criterion,device):
    print("Testing the model on testing dataset .. ")
    test_loss = 0
    accuracy = 0
    if device=='cuda':
       devicee=torch.device("cuda:0")
       model.to(devicee)
    model.eval()

    for images, labels in testloader:

        
        if device=='cuda':
           images, labels = images.to(devicee), labels.to(devicee)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print("Test Loss: {} , Accuracy : {}".format(test_loss/len(testloader),accuracy/len(testloader)))