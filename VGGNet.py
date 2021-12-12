import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import random
import time
import numpy as np
import copy
import random
import time

__author__='Research group of Huanyang Chen'

'''
VGGNet code. We used VGG16 for training. More details can see .pdf
'''

SEED=1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

class VGG(nn.Module):
    def __init__(self,features, output_dim):
        super().__init__()
        self.features=features
        self.avgpool=nn.AdaptiveAvgPool2d(7)
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.Dropout(0.5),
            nn.Linear(4096,output_dim),
        )
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        h=x.view(x.shape[0],-1)
        x=self.classifier(h)
        return x,h
vgg11_config=[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
vgg13_config=[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']
vgg16_config=[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
vgg19_config=[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']

def get_vgg_layers(config,batch_norm):
    layers=[]
    in_channels=3
    for c in config:
        assert c == 'M' or isinstance(c,int)
        if c=='M':
            layers+=[nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d=nn.Conv2d(in_channels,c,kernel_size=3,padding=1)
            if batch_norm:
                layers+=[conv2d,nn.BatchNorm2d(c),nn.ReLU(inplace=True)]
            else:
                layers+=[conv2d,nn.ReLU(inplace=True)]
            in_channels=c
    return nn.Sequential(*layers)

vgg16_layers=get_vgg_layers(vgg16_config,batch_norm=True)
OUTPUT_DIM=10
model=VGG(vgg16_layers,OUTPUT_DIM)
pretrained_model=models.vgg16_bn(pretrained=True)
IN_FEATURES=pretrained_model.classifier[-1].in_features
final_fc=nn.Linear(IN_FEATURES,OUTPUT_DIM)
# Build the VGG16 model that has been trained in ImageNet dataset.
# The retrain it using trainsfer learning.
pretrained_model.classifier[-1]=final_fc
model.load_state_dict(pretrained_model.state_dict(),False)

def count_parameters(model):
    # Returns the parameters that need to be optimized.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The model has %d trainable parameters'%(count_parameters(model)))

pretrained_size=224
pretrained_means=[0.485,0.456,0.406]
pretrained_stds=[0.229,0.224,0.225]

# Preprocessing for training sets.
# To enhance the generalization ability.
train_transforms=transforms.Compose([ 
    transforms.Resize([pretrained_size,pretrained_size]),
    transforms.RandomRotation(5), 
    transforms.RandomHorizontalFlip(0.5), 
    transforms.RandomCrop(pretrained_size,padding=10), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=pretrained_means,std=pretrained_stds)
])

# Preprocessing for validation sets and test sets.
test_transforms=transforms.Compose([
    transforms.Resize([pretrained_size,pretrained_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,std=pretrained_stds)
 ])

path_train=r"DataProcessing\DataSource\OMDRdatasets\train"
train_data=datasets.ImageFolder(path_train,train_transforms)
path_test=r"DataProcessing\DataSource\OMDRdatasets\test"
test_data=datasets.ImageFolder(path_test,test_transforms)
VALID_RATIO = 0.9
n_train_examples = int(len(train_data)*VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = data.random_split(train_data,[n_train_examples, n_valid_examples])
valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms
print('Number of training examples:%s' %(len(train_data)))
print('Number of validation examples:%s' %(len(valid_data)))
print('Number of testing examples:%s'%(len(test_data)))

def normalize_image(image):
    image_min=image.min()
    image_max=image.max()
    image.clamp_(min=image_min,max=image_max)
    image.add_(-image_min).div_(image_max-image_min+1e-5)
    return image

classes=test_data.classes
BATCH_SIZE=64
train_iterator=data.DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
valid_iterator=data.DataLoader(valid_data,batch_size=BATCH_SIZE)
test_iterator=data.DataLoader(test_data,batch_size=BATCH_SIZE)
device=torch.device('cpu')
criterion=nn.CrossEntropyLoss()   # loss function
model=model.to(device)
criterion=criterion.to(device)

class IteratorWrapper:
    # jump to the next batch
    def __init__(self,iterator):
        self.iterator=iterator
        self._iterator=iter(iterator)

    def __next__(self):
        try:
            inputs,labels=next(self._iterator)
        except StopIteration:
            self._iterator=iter(self.iterator)
            inputs,labels=next(self._iterator)
        return inputs,labels

    def get_batch(self):
        return next(self)
FOUND_LR=5e-04
params=[
    {'params':model.features.parameters(),'lr':FOUND_LR/10},
    {'params':model.classifier.parameters()}
]
optimizer=optim.Adam(params,lr=FOUND_LR)

def calculate_accuracy(y_pred,y):
    top_pred=y_pred.argmax(1,keepdim=True)
    correct=top_pred.eq(y.view_as(top_pred)).sum()
    acc=correct.float()/y.shape[0]
    return acc

def train(model,iterator,optimizer,criterion,device):
    epoch_loss=0
    epoch_acc=0
    # To enable the Batch Normalization and Dropout
    model.train()
    for (x,y) in iterator:
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_pred,_=model(x)
        loss=criterion(y_pred,y)
        acc=calculate_accuracy(y_pred,y)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        epoch_acc+=acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

def evaluate(model,iterator,criterion,device):
    epoch_loss=0
    epoch_acc=0
    model.eval()
    with torch.no_grad():
        for (x,y) in iterator:
            x=x.to(device)
            y=y.to(device)
            y_pred,_=model(x)
            loss=criterion(y_pred,y)
            acc=calculate_accuracy(y_pred,y)
            epoch_loss+=loss.item()
            epoch_acc+=acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

def epoch_time(start_time,end_time):
    elapsed_time=end_time-start_time
    elapsed_mins=int(elapsed_time/60)
    elapsed_secs=int(elapsed_time-(elapsed_mins*60))
    return elapsed_mins,elapsed_secs
EPOCHS=10
best_valid_loss=float('inf')

for epoch in range(EPOCHS):
    start_time=time.monotonic() 
    train_loss,train_acc=train(model,train_iterator,optimizer,criterion,device)
    valid_loss,valid_acc=evaluate(model,valid_iterator,criterion,device)
    if valid_loss<best_valid_loss:
        best_valid_loss=valid_loss
        torch.save(model.state_dict(),'trained-model.pt')    # Save the trained VGGNet model.
    print('Epoch: %s' %(epoch+1))
    print('\tTrain Loss: %s|Train Acc: %s' %(train_loss,train_acc))
    print('\t Val. Loss: %s|Val. Acc:%s' %(valid_loss,valid_acc))
end_time=time.monotonic()
epoch_mins,epoch_secs=epoch_time(start_time,end_time)
print('Epoch Time: %sm %ss' %(epoch_mins,epoch_secs))    # train time
model.load_state_dict(torch.load('trained-model.pt'),False)

start_time=time.monotonic()
test_loss,test_acc=evaluate(model,test_iterator,criterion,device)
end_time=time.monotonic()
epoch_mins,epoch_secs=epoch_time(start_time,end_time)
print('epoch_mins:%s,epoch_secs:%s'%(epoch_mins,epoch_secs))   # test time
print('Test Loss:%s|Test Acc:%s '%(test_loss,test_acc))
def get_predictions(model,iterator):
    model.eval()
    images=[]
    labels=[]
    probs=[]  # the probability of each category
    with torch.no_grad():
        for (x,y) in iterator:
            x=x.to(device)
            y_pred,_=model(x)
            y_prob=F.softmax(y_pred,dim=-1)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())
    images=torch.cat(images,dim=0)
    labels=torch.cat(labels,dim=0)
    probs=torch.cat(probs,dim=0)
    return images,labels,probs
images,labels,probs=get_predictions(model,test_iterator)
pred_labels=torch.argmax(probs,1)  # The predicted hand gesture.

