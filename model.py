import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from tv_transform_loader import transform_load



def model_to_use(architecture, hidden_layers):
    """Defines the model to train on.
       Arguments: architecture  - ImageNet model to use (string)
                  hidden_layers - number of input nodes for classifier portion of the model chosen (string)
       Output: pre-trained model as specified
    """
    arch = architecture
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Create ordered dictionary defining the new classifier with hidden layer input
    hidden_layer_input = hidden_layers
    classifier = nn.Sequential(OrderedDict([
    ('0', nn.Linear(hidden_layer_input, 1200)),
    ('1', nn.ReLU()),
    ('2', nn.Dropout(p=0.4)),
    ('3', nn.Linear(1200, 102)),
    ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model



def train_model(dir_path, architecture, hidden_layers, dev, learning_rate, epochs, save_dir):
    """Utilizes GPU if specified.  Defines the hyperparameters to tune the model with and trains the model accordingly.
       Arguments: dir_path      - path to the image folder to be classified (string)
                  architecture  - as required by model_to_use (string)
                  hidden_layers - as required by model_to_use (string)
                  dev           - either Y (gpu) or N (cpu) (string)
                  learning_rate - factor to adjust gradient descent step by, when backpropagating the optimizer error (string)
                  epochs        - number of times the model is utilizing/'training' on the training dataset (string)
                  save_dir      - location of directory to save checkpoint file (string)
       Output: cl_checkpoint.pth: saved checkpoint file
    """   
    if dev == 'Y':
        try:
            device = torch.device("cuda:0")
        except:
            device = torch.device("cpu")
        
    learn_rate = learning_rate    
    model = model_to_use(architecture, hidden_layers) 
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learn_rate)
    model.to(device);    
    
    ep = epochs 
    
    steps = 0
    train_loss = 0
    print_every = 50
    t_loader, v_loader = transform_load(dir_path)[0:2]
    
    for epoch in range(ep):
        for inputs, labels in t_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in v_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        validation_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                # Track the loss and accuracy on the validation set to determine the best hyperparameters        
                print(f"Epoch {epoch+1}/{epochs}..\t"
                      f"Train loss: {train_loss/print_every:.3f}..\t\t"
                      f"Validation loss: {validation_loss/len(v_loader):.3f}..\t"
                      f"Validation accuracy: {accuracy/len(v_loader):.3f}")
                train_loss = 0
                model.train()
        
        
        # Get the training dataset and class to idx for checkpoint file
        model.class_to_idx = transform_load(dir_path)[-1].class_to_idx
        checkpoint = {'epochs': ep,
              'arch': architecture,
              'hidden_units': hidden_layers,   
              'learning_rate': learn_rate,
              'model_state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, save_dir + '/cl_checkpoint.pth')
                