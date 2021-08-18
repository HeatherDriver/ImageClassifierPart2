import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

def process_image(path_to_image):
    ''' Scales, crops, and normalizes a PIL image for use within a PyTorch model
        Arguments: path_to_image  - Name of image to process (string)
        Output: Torch Tensor for input to model
    '''
    image = path_to_image[0]
    means = np.array([[0.485, 0.456, 0.406]])
    std_d = np.array(([0.229, 0.224, 0.225]))
    
    with Image.open(image) as im:
        # resizing the images where the shortest side is 256 pixels, keeping the aspect ratio. 
        if im.size[0] < im.size[1]:
            size = 256, im.size[1]
        else:
            size = im.size[0], 256
        im.thumbnail(size)
        
        # crop the centre 224 pixel portion
        im_crop = im.crop((0, 0, 224, 224))
        
        # convert to numpy array, standardize and normalize it through array broadcasting
        np_image = (np.array(im_crop)) / 255
        np_std = (np_image - means) / std_d
        
        # re-order dimensions
        np_std = np_std.transpose((2,0,1))
        inputs = torch.from_numpy(np_std) 
        inputs = inputs.unsqueeze(0)
        inputs = inputs.float()
        return inputs

def image_cat_to_name(filen):
    ''' Loads json file for use with model's category output
        Arguments: filen  - Name of json file to load
        Output: Loaded json file
    '''    
    with open(filen, 'r') as f:
        return json.load(f)
    