"""
This script will train a new network on a given dataset of images using transfer learning
with a pre-trained model. It will then save the newly trained model as a checkpoint.

Required input:     - data_dir: data directory with image data sets (train and validation sets in separate folders,
                      root/train and root/valid)

Optional inputs:    - save_dir: directory in which to save the model checkpoint after training (default is data directory)
                    - arch: pre-trained model architecture (default is vgg19)
                    - learning_rate: type float, learning rate for optimizer (default is 0.001)
                    - hidden_units: type int, number of units in hidden layer (default is 2500)
                    - epochs: type int, number of epochs for training (default is 3)
                    - gpu: whether to train on gpu (defailt is cpu)
"""

# set up command line inputs
import argparse

parser = argparse.ArgumentParser(description='Trains new network on image dataset and saves network to checkpoint')
parser.add_argument('data_dir', type=str)
parser.add_argument('--save_dir', help='Save model checkpoints in given directory', type=str)
parser.add_argument('--arch', help='Choose pre-trained VGG model architecture', type=str, default='vgg19')
parser.add_argument('--learning_rate', help='Set learning learning rate', type=float, default=0.001)
parser.add_argument('--hidden_units', help='Set number of hidden units', type=int, default=2500)
parser.add_argument('--epochs', help='Set number of epochs', type=int, default=3)
parser.add_argument('--gpu', help='Use GPU for training instead of CPU', action='store_true')
args = vars(parser.parse_args())

# import other packages
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session
from utils_functions import load_data
from model_functions import validation, new_classifier, save_checkpoint

from collections import OrderedDict

# set input args as variables
data_dir = args['data_dir']
arch = args['arch']
lr = args['learning_rate']
hidden_units = args['hidden_units']
epochs = args['epochs']
device = torch.device("cuda:0" if args['gpu'] is True else "cpu")

# load datasets and get trainloader and validloader
train_dataset, valid_dataset, trainloader, validloader = load_data(data_dir)

# load a pretrained network
if arch == "vgg16":
    model = models.vgg16(pretrained=True)
if arch == "vgg13":
    model = models.vgg13(pretrained=True)
if arch == "vgg11":
    model = models.vgg11(pretrained=True)
if arch == "vgg19" or arch == None:
    model = models.vgg19(pretrained=True)

# freeze parameters to prevent backpropagation with them
for param in model.parameters():
    param.requires_grad = False

# define new classifier to replace current model classifier
input_size = 25088
output_size = 102

model.classifier = new_classifier(input_size, hidden_units, output_size)

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


# set class_to_idx mapping for model
model.class_to_idx = train_dataset.class_to_idx

# save model checkpoint
if args['save_dir'] is None:
    save_filepath = data_dir + '/checkpoint.pth'
else:
    save_filepath = args['save_dir'] + '/checkpoint.pth'

save_checkpoint(model, arch, optimizer, epochs, input_size, output_size, hidden_units, save_filepath)
