import torch
from torchvision import datasets, transforms
import os

def get_sub_folder_names(dir_path):
    """Function to return the sub-folder contents of data_dir.
           Args:
            dir_path: Directory with Training and Validation folders
           Returns:
            List of sub-folder names
        """
    folders = os.listdir(os.getcwd() + '/' + str(dir_path[0]))
    paths = [os.getcwd() + '/' + str(dir_path[0]) + '/' + folder for folder in folders]
    return folders, paths


    
def transform_load(dir_path):
    """Function to apply appropriate training-specific transformations to the training folder,
       then load the data.
       Args:
            dir_path: Folder location (string)
       Returns:
            Data loaded to PyTorch's dataloader
        """
    folders, paths = get_sub_folder_names(dir_path)
    for folder, path in zip(folders, paths):
        if folder == 'train':
            t_transform = transforms.Compose([transforms.RandomRotation(15),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
            t_imagef = datasets.ImageFolder(path, transform=t_transform)
            t_loader = torch.utils.data.DataLoader(t_imagef, batch_size=64, shuffle=True)
        if folder == 'valid':
            v_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
            v_imagef = datasets.ImageFolder(path, transform=v_transform)
            v_loader = torch.utils.data.DataLoader(v_imagef, batch_size=64, shuffle=True)
    return t_loader, v_loader, t_imagef