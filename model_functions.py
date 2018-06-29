# import packages
import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict


def validation(model, criterion, dataloader):
    """
    Takes in a model, criterion, and validation dataloader as inputs 
    and calculates the loss and accuracy on the validation set.

    Inputs: model, criterion, dataloader
    Outputs: valid_loss, accuracy
    """
    valid_loss = 0
    accuracy = 0
    for images, labels in dataloader:
        # move inputs to cuda
        images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        
        loss = criterion(output, labels)
        valid_loss += loss.item()
        
        ps = torch.exp(output)
        predicted_label = ps.max(dim=1)[1]
        
        # compare output data with true labels (1 where equal, 0 where unequal)
        equality = (labels.data == predicted_label)
        # convert from byte tensor to float tensor to be able to take mean
        equality = equality.type(torch.FloatTensor)
        
        accuracy += equality.mean()
        
    return valid_loss, accuracy


def save_checkpoint(model, arch, optimizer, epochs, input_size, output_size, hidden_units, filename):
    """
    Saves model state and architecture to checkpoint under given filename.
    """
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layer': hidden_units,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'arch': arch
                 }

    torch.save(checkpoint, filename)


def load_checkpoint(filepath):
    """
    Loads checkpoint from given filepath and sets new model to the checkpoint states and architecture.
    Returns the model.
    """
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    
    # load a pretrained network
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    if arch == "vgg19" or arch == None:
        model = models.vgg19(pretrained=True)
    
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_layer']
    output_size = checkpoint['output_size']

    # build new classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_size)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(p=0.5)),
                                            ('output', nn.Linear(hidden_size, output_size)),
                                            ('logSoftmax', nn.LogSoftmax(dim=1))
                                           ]))
    
    # replace model classifier with new classifier
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    # load optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def new_classifier(input_size, hidden_size, output_size):
    """
    Creates new classifier with one hidden layer and an output given by a logSoftmax function
    """
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_size)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(p=0.5)),
                                            ('output', nn.Linear(hidden_size, output_size)),
                                            ('logSoftmax', nn.LogSoftmax(dim=1))
                                          ]))
    return classifier
