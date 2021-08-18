from train_input_args import get_train_input_arguments
from model import *

def main():
    # Get input arguments
    in_arg = get_train_input_arguments()
    # Train model and create checkpoint file
    trained_model = train_model(in_arg.directory, in_arg.arch, in_arg.hidden_units, 
                                in_arg.gpu, in_arg.learning_rate, in_arg.epochs, in_arg.save_dir)
if __name__ == "__main__":
    main()
