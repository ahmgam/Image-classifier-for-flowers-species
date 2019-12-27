import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from input_args import get_input_args
from checkpoints_ops import load_chk
from processing_img import process_image
import json

in_args=get_input_args('predict')
model,epochs,optimizer=load_chk(in_args.chk_dir)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device=torch.device("cuda:0" if in_args.device=='cuda' else "cpu")
    image=torch.from_numpy(process_image(image_path)).float()
    image.to(device)
    image = image.unsqueeze(0)
    output = model.forward(image)
    outp=torch.exp(output)
    predicted,indeces = outp.topk(topk)
    with open(in_args.category_names, 'r') as f:
        classes = json.load(f)
    #out =[topk]
    topnames=[None] * topk
    for i in range(0,topk,1):
        topnames[i]=classes[str(indeces.tolist()[0][i])]
    #for j in range(0,4,1):
     # out[j]=classes[indeces[j]]
    return predicted[0].tolist(),topnames
    
# TODO: Implement the code to predict the class from an image file
probs, classes = predict(in_args.input_dir, model,in_args.top_k)
print(probs)
print(classes)