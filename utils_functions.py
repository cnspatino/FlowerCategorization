# import packages
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

def load_data(data_dir):
    """
    Loads datasets from the given data directory and performs the necessary transforms to produce
    dataloaders for the training and validation sets.
    
    Input: data directory that contains a 'train' and 'valid' folder with png images like follows:
                training set: data_dir/train/sample.png
                validation set: data_dir/valid/sample.png
    Returns: trainloader and validloader to be used for model training
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # define transforms for training and validation sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(40),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229,0.224,0.225])
                                      ])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229,0.224,0.225])
                                     ])

    # load the datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root = train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(root = valid_dir, transform = valid_transforms)

    # define the dataloaders using the image datasets and transforms
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 50, shuffle=True)
    
    return train_dataset, valid_dataset, trainloader, validloader


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.
        Returns a Numpy array of image.
    '''
    
    # load image and get size
    im = Image.open(image)
    width, height = im.size
    
    
    # resize image so shortest side is 256 pixels
    if width < height:
        im.thumbnail((255, height))
    else:
        im.thumbnail((width, 255))
    
    # get new width and height to define crop box
    width, height = im.size
    crop_box = ((width - 224)/2, (height - 224)/2, (width + 224)/2, (height + 224)/2)
    
    # crop image in center
    im = im.crop(crop_box)
    
    # convert image to numpy array and convert values to floats 0 to 1
    np_image = np.array(im)
    np_image = np_image/255
    
    # normalize the color channels
    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - norm_mean) / norm_std
    
    # transpose image array so that color channel is first dimension
    processed_image = np_image.transpose((2,0,1))

    return processed_image
