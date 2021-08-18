# Import Argparse module
import argparse

def get_train_input_arguments():
    """
    Retreives command line arguments as provided by the user when running the train.py program
    from the terminal. 
    Parameters:
     None
    Returns:
     parse_args() - stores the arguments
     """

    parser = argparse.ArgumentParser(description='Provide directory for images')
    
    # 1: Choose data directory
    parser.add_argument('directory', metavar='N', type = str, nargs='+', help = 'directory for images for model')
    
    # 2: Set directory to save checkpoints
    parser.add_argument('--save_dir', type = str, default = 'checkpoints_save/', 
                        help = 'Path to directory to save checkpoint file')
    
    # 3: Choose architecture
    parser.add_argument('--arch', type = str, default = 'vgg11', help = 'Select model architecture')
    
    # 4.1: Set hyperparameters - learning rate
    parser.add_argument('--learning_rate', type = float, default = '0.01', help = 'Select learning rate')
    
    # 4.2: Set hyperparameters - hidden units
    parser.add_argument('--hidden_units', type = int, default = '25088', help = 'Select hidden units')
    
    # 4.3: Set hyperparameters - epochs
    parser.add_argument('--epochs', type = int, default = '8', help = 'Select number of epochs')
    
    # 5: Use GPU for training
    parser.add_argument('--gpu', type = str, default = 'Y', help = 'Select to execute on GPU (Y/N)')
    
    return parser.parse_args()
    