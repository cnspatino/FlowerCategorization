# ImageClassification-neural-network
Command line programs that train an image classifier on a dataset using deep learning with PyTorch and predict the labels of new images using the trained model. 

- train.py: This script will train a new neural network on a given dataset of images using transfer learning
with a [pre-trained model](https://pytorch.org/docs/stable/torchvision/models.html). It will then save the newly trained model as a checkpoint.
  - Required input:     
    - data_dir: data directory with image data sets (train and validation sets in separate folders, root/train and root/valid)

  - Optional inputs:    
    - save_dir: directory in which to save the model checkpoint after training (default is data directory)
    - arch: type str, pre-trained model architecture of vgg or densenet (default is vgg19)
    - learning_rate: type float, learning rate for optimizer (default is 0.001)
    - hidden_units: type int, number of units in hidden layer (default is 2500)
    - epochs: type int, number of epochs for training (default is 3)
    - gpu: whether to train on gpu (defailt is cpu)
                    
- predict.py: This script loads a trained network from a checkpoint file and uses the model to 
predict the class for an input image. It returns the predicted class or category along with 
the probability.
  - Required Inputs:    
    - image_path: type str, filepath to single image
    - checkpoint: type str, checkpoint filepath for trained network
  - Optional inputs:    
    - topk: type int, returns the top K most likely classes (default is 1)
    - category_names: type str, json file to map classes to real names
    - gpu: whether to train on gpu (default is cpu)
                    
- model_functions.py: Contains functions related to the model.
    - validation function: Takes in model, criterion, and validation dataloader as inputs 
      and calculates the loss and accuracy on the validation set.
    - save_checkpoint function: Saves model state and architecture to checkpoint under given filename.
    - load_checkpoint function: Loads checkpoint from given filepath and sets new model to the checkpoint states and        
      architecture.
    - new_classifier function: Creates new classifier with one hidden layer and an output given by a logSoftmax function

- utils_functions.py: Contains utility functions for loading data and preprocessing images.
    - load_data function: Loads datasets from the given data directory and performs the necessary transforms to produce
      dataloaders for the training and validation sets.
    - process_image function: Scales, crops, and normalizes a PIL image for a PyTorch model. Returns a Numpy array of the   
      image.
