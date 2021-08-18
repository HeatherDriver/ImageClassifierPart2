from predict_input_args import get_predict_input_arguments
from load_checkpoint_predict import load_checkp_predict
from image_processing import *


def main():
    # Get input arguments
    in_arg = get_predict_input_arguments()

    # Load checkpoint file and predict on processed image
    probs, classes = load_checkp_predict(in_arg.checkpoint, in_arg.directory,  in_arg.gpu, in_arg.top_k)
    classes_dict = image_cat_to_name(in_arg.category_names)
    classes = [classes_dict[c] for c in classes]
    # Print classification and probabilities
    for c, p in zip(classes, probs):
        print(c, '\t:\t', p)
    
if __name__ == "__main__":
    main()
    