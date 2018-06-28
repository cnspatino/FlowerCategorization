"""
This script will train a new network on a given dataset of images using transfer learning
with a pre-trained model. It will then save the newly trained model as a checkpoint.

Input: data directory with image data sets (train and validation sets in separate folders)
Optional inputs:    - save directory (if different from data directory)
                    - pre-trained model architecture (default is vgg19)
                    - learning rate (default is 0.001)
                    - number of units in hidden layer (default is 2500)
                    - number of epochs (default is 3)
                    - whether to train on gpu (defailt is cpu)

"""

# set up command line inputs
import argparse

parser = argparse.ArgumentParser(description='Trains new network on image dataset and saves network to checkpoint')
parser.add_argument('data_dir')
parser.add_argument('-s', '--save_dir', help='Save model checkpoints in given directory')
parser.add_argument('-a','--arch', help='Choose pre-trained model architecture', type=str, default='vgg19')
parser.add_argument('-lr', '--learning_rate', help='Set learning learning rate', type=int, default=0.001)
parser.add_argument('-hu', '--hidden_units', help='Set number of hidden units', type=int, default=2500)
parser.add_argument('-e', '--epochs', help='Set number of epochs', type=int, default=3)
parser.add_argument('-g', '--gpu', help='Use GPU for training instead of CPU', action='store_true')
args = vars(parser.parse_args())

# import other packages
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from workspace_utils import active_session
from utils_functions import load_data
from model_functions import validation, new_classifier, save_checkpoint

from collections import OrderedDict

# set input args as variables
data_dir = args.data_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = epochs
device = torch.device("cuda:0" if args.gpu is True else "cpu")

# load datasets and get trainloader and validloader
train_dataset, valid_dataset, trainloader, validloader = load_data(data_dir)

# load a pretrained network
model = models.arch(pretrained=True)

# freeze parameters to prevent backpropagation with them
for param in model.parameters():
    param.requires_grad = False

# define new classifier to replace current model classifier
input_size = 25088
hidden_size = hidden_units
output_size = 102

model.classifier = new_classifier(input_size, hidden_size, output_size)

# set criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

steps = 0
running_loss = 0
print_every = 10

# move to cuda to use gpu
model.to(device)

with active_session():
    for e in range(epochs):
    # make sure model is in train mode
        model.train()

        for images, labels in trainloader:
            # move inputs to cuda
            images, labels = images.to(device), labels.to(device)
        
            steps += 1
        
            # clear the gradients
            optimizer.zero_grad()
        
            #forward pass
            output = model.forward(images)
            # calculate loss and backpropagate
            loss = criterion(output, labels)
            loss.backward()
            # update the parameters using the optimizer
            optimizer.step()
        
            running_loss += loss
        
            if steps % print_every == 0:
            
                # change model to evaluation mode
                model.eval()
            
                # perform validation steps
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, criterion, validloader)
            
                print('Epoch {}/{}'.format(e+1, epochs))
                print('Training loss: {:.4f}'.format(running_loss/print_every))
                print('Validation loss: {:.4f}'.format(valid_loss/len(validloader)))
                print('Validation accuracy: {:.4f}%'.format((accuracy/len(validloader))*100))
            
                running_loss = 0
            
                # change model back to training mode
                model.train()


# save checkpoint of current model
model.class_to_idx = train_dataset.class_to_idx

checkpoint_state = {'input_size': input_size,
                    'output_size': output_size,
                    'hidden_layer': hidden_size,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epochs': epochs,
                    'class_to_idx': model.class_to_idx
                   }

if save_dir is not None:
    save_filepath = save_dir + '/checkpoint.pth'
else:
    save_filepath = data_dir + 'checkpoint.pth'

save_checkpoint(checkpoint_state, save_filepath)
