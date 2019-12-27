from PIL import Image
import numpy as np
def process_image(imagedir):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image=Image.open(imagedir)
    print("loaded successfully")
    print(type(image))
    # Resize
    if image.size[0] > image.size[1]:
        image=image.resize((image.size[0]+2, 256))
    else:
        image=image.resize((256, image.size[1]+2))
        
    
    # Crop 
    width, height = image.size
    left = (width-224)/2
    bottom = (height-224)/2
    right = left + 224
    top = bottom + 224
    
    image = image.crop((left, bottom, right, top))
    image=image.convert('RGB')
    # Normalize
    nimage = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nimage = (nimage - mean) / std
    nimage = nimage.transpose((2, 0, 1))
    return nimage