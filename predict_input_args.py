# Import Argparse module
import argparse

def get_predict_input_arguments():
    """
    Retreives command line arguments as provided by the user when running the predict.py program
    from the terminal. 
    Parameters:
     None
    Returns:
     parse_args() -stores the arguments
     """

    parser = argparse.ArgumentParser(description='Provide directory for images')
    
    # 1: Choose path to image
    parser.add_argument('directory', metavar='N', type = str, nargs='+', help = 'path to image to predict')
    
    # 2: Checkpoint file to use
    parser.add_argument('checkpoint', metavar='N', type = str, nargs='+', help = 'checkpoint file to load')
    
    # 3: Choose top k number of class predictions
    parser.add_argument('--top_k', type = int, default = '5', help = 'Select top k prediction classes')
    
    # 4: Use mapping of categories to real names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Select file mapping flower number to the name')
    
    # 5: Use GPU for predicting
    parser.add_argument('--gpu', type = str, default = 'Y', help = 'Select to execute on GPU (Y/N)')
    
    return parser.parse_args()
    