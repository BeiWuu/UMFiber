import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random

__author__='Research group of Huanyang Chen'

'''
Use the trained model to recognize new gestures.
'''

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        
        self.features = features
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3
    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c
    return nn.Sequential(*layers)
vgg16_layers = get_vgg_layers(vgg16_config, batch_norm = True)

OUTPUT_DIM = 10
model = VGG(vgg16_layers, OUTPUT_DIM)

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])
path_test=r"DataProcessing\DataSource\OMDRdatasets\test"   # Store the optical microfiber dynamic response in this folder.
test_data=datasets.ImageFolder(path_test,test_transforms)
BATCH_SIZE = 1
test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)
model.load_state_dict(torch.load('trained-model.pt', map_location='cpu'),False)

def get_predictions(model, iterator):
    # Output the ground truth and predicted label.
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            y_pred, _ = model(x)
            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)
            correct=top_pred.eq(y.view_as(top_pred))
            '''
            # Output the incorrect sample.
            if correct==False:
                print('Ground Truth',y,'Predicted',top_pred)
            '''
            # Output the predicted and true label.
            print('Ground Truth',y,'Predicted',top_pred)

get_predictions(model, test_iterator)
