import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import time
import numpy as np
import copy
import random
import time

__author__='Research group of Huanyang Chen'

'''
Shallow CNNs code.
It contains two convolutional layers, which are both followed by a ReLU activation function and a max-pooling layer, and 10 categories are output after passing through two fully connected layers.
'''

SEED=1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

class Cnn(nn.Module):
    def __init__(self, in_dim, n_class=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=12))
        self.fc = nn.Sequential(
            nn.Linear(1296,360),
            nn.Linear(360, n_class))
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
model = Cnn(in_dim=3, n_class=10)

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]
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

BATCH_SIZE=64
train_iterator=data.DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE)
valid_iterator=data.DataLoader(valid_data,batch_size=BATCH_SIZE)
test_iterator=data.DataLoader(test_data,batch_size=BATCH_SIZE)
START_LR=1e-07
optimizer=optim.Adam(model.parameters(),lr=START_LR)
device=torch.device('cpu')
criterion=nn.CrossEntropyLoss()
model=model.to(device)
criterion=criterion.to(device)

FOUND_LR=5e-04
params=[
    {'params':model.parameters(),'lr':FOUND_LR/10},
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
    model.train()  # To enable the Batch Normalization and Dropout
    for (x,y) in iterator:
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_pred=model(x)
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
            y_pred=model(x)
            loss=criterion(y_pred,y)
            acc=calculate_accuracy(y_pred,y)
            epoch_loss+=loss.item()
            epoch_acc+=acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

def epoch_time(start_time,end_time):   # Get running time.
    elapsed_time=end_time-start_time
    elapsed_mins=int(elapsed_time/60)
    elapsed_secs=int(elapsed_time-(elapsed_mins*60))
    return elapsed_mins,elapsed_secs

EPOCHS=10      # Train at 10 epochs.
best_valid_loss=float('inf')
for epoch in range(EPOCHS):
    start_time=time.monotonic()
    train_loss,train_acc=train(model,train_iterator,optimizer,criterion,device)
    valid_loss,valid_acc=evaluate(model,valid_iterator,criterion,device)
    if valid_loss<best_valid_loss:
        best_valid_loss=valid_loss
        torch.save(model.state_dict(),'trained-cnn-model.pt')  # Save the trained VGGNet model.
    print('Epoch: %s' %(epoch+1))
    print('\tTrain Loss: %s|Train Acc: %s' %(train_loss,train_acc))
    print('\t Val. Loss: %s|Val. Acc:%s' %(valid_loss,valid_acc))
end_time=time.monotonic()
epoch_mins,epoch_secs=epoch_time(start_time,end_time)
print('Epoch Time: %sm %ss' %(epoch_mins,epoch_secs))
model.load_state_dict(torch.load('trained-model.pt'),False)

start_time=time.monotonic()
test_loss,test_acc=evaluate(model,test_iterator,criterion,device)
end_time=time.monotonic()
epoch_mins,epoch_secs=epoch_time(start_time,end_time)
print('epoch_mins:%s,epoch_secs:%s'%(epoch_mins,epoch_secs))
print('Test Loss:%s|Test Acc:%s '%(test_loss,test_acc))
