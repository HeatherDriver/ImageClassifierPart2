import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from image_processing import process_image
import os


def load_checkp_predict(checkpoint_filen, path_to_image, dev, topk):
    ''' Loads appropriate checkpoint file and then executes model prediction on image
        Arguments: checkpoint_filen  - Name of checkpoint file to load
                   path_to_image     - Path of image to use in prediction model
                   dev               - Device to use - "Y" is gpu/cuda and "N" is cpu
                   top_k             - Return top k number of classes and probabilities
        Output: Probabilities and classifications as predicted by model
    '''       
    filepath = os.getcwd() + '/' + 'checkpoints_save/' + str(checkpoint_filen[0])
    checkpoint = torch.load(filepath)
    
    epoch = checkpoint['epochs']
    learnrate = checkpoint['learning_rate']
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    
    model = getattr(models, arch)(pretrained=True)
            
    classifier = nn.Sequential(nn.Linear(hidden_units, 1200),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(1200, 102),
                          nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.SGD(model.classifier.parameters(), lr=learnrate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    if dev == 'Y':
        try:
            model.to("cuda")
        except:
            model
   
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    inputs = process_image(path_to_image)
    
    with torch.no_grad():
        model.eval()
        out = model.forward(inputs.cuda())
        ps = torch.exp(out)
        top_p, top_class = ps.topk(topk, dim=1)
        
        top_prob_array = top_p.cpu()
        top_prob_array = top_prob_array.numpy()
        probs = [t for top_prob in top_prob_array for t in top_prob]
        
        top_class_array = top_class.cpu()
        top_class_array = top_class_array.numpy()
        classes = [model.idx_to_class[t] for top_class in top_class_array for t in top_class]

    return probs, classes
