import argparse

parser = argparse.ArgumentParser(description='Predicts flower name from an image')
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('-tk', '--topk', help='Return top K most likely classes', type=int, default=5)
parser.add_argument('-c', '--category_names', help='Map categories to real names', type=str)
parser.add_argument('g', '--gpu', help='Use GPU for inference instead of CPU', action='store_true')
args = vars(parser.parse_args())

# import other packages
import numpy as np
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

from workspace_utils import active_session
from model_functions import load_checkpoint
from utils_functions import process_image

from collections import OrderedDict

from PIL import Image
    
# set input args to variables
image_path = args.image_path
checkpoint = args.checkpoint
topk = args.topk
category_map = args.category_names
device = torch.device("cuda:0" if args.gpu is True else "cpu")

model = load_checkpoint(checkpoint)
    
# move model to cuda and use in eval mode
model.to(device)
model.eval()
    
#load and process image and convert to torch tensor
processed_image = process_image(image_path)
image = torch.from_numpy(processed_image)
    
# add additional dimension to torch tensor to match input size - should be (1,3,224,224)
image = image.unsqueeze(0)
image = image.to(device)
 
# predict classification using the trained model (output will be from softmax function)
with torch.no_grad():
    output = model.forward(image.float())
    
# calculate class probablities by taking the exponent of the output
ps = torch.exp(output)
    
# get top k probabilities and indices
probs, indices = ps.topk(topk)
    
# load dictionary for mapping indices to classes and invert
class_to_idx = model.class_to_idx
idx_to_class = {value:key for key, value in class_to_idx.items()}
    
# get topk classes from indices
topk_predictions = []
indices = list(indices.cpu().numpy()[0])
for i in indices:
    topk_predictions.append(idx_to_class[i])

topk_probs = probs.cpu().numpy()[0]

# get top prediction probablity
prediction_prob = topk_probs[0]

# get top prediction class or name
if args.category_names is None:
    prediction = topk_predictions[0]
else:
    with open(category_map, 'r') as f:
        cat_to_name = json.load(f)
    
    names = []
    for c in classes:
        names.append(cat_to_name[c])
    
    topk_predictions = names
    prediction = topk_predictions[0]

if topk is None:
   print(prediction)
   print(prediction_prob)
   return prediction, prediction_prob
       
else:
    print(topk_predictions)
    print(topk_probs)
    return topk_predictions, topk_probs

