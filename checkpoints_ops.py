
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from collections import OrderedDict

def save_chk (model,optimizer,epochs,chk_dir,hidden,insize,arch,lrn):
    checkpoint = {'arch': arch,
                  'input_size': insize,
                  'output_size': 102,
                  'hidden_layers': hidden,
                  'model_state_dict': model.state_dict(),
                  'opt_state_dict':optimizer.state_dict(),
                  'epochs':epochs,
                  'learn_rate':lrn
                 }
    
    torch.save(checkpoint, chk_dir)
    print("Model saved.")

def load_chk (filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg':
        model = models.vgg16(pretrained=True)
        
    if  checkpoint['arch'] == 'alexnet'  :
        model = models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
   
        
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout()),
                          ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))           
                          ]))
    
    epochs=checkpoint['epochs']
    model.classifier=classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer=optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    return model,epochs,optimizer
  
