import os 
import random
from shutil import copy2

__author__='Research group of Huanyang Chen'

'''
Divide the picture into training sets and test sets
Noting: The division of training and validation sets are carried out in vgg.py
'''

dir=r"DataSource\picFile" 
list1=os.listdir(dir)   # Different gesture files
data_capacity=1   # Determine the sample size of the dataset, it can be (0.1, 0.2, ..., 1)
for i in list1:
    list2=os.listdir(os.path.join(dir,str(i)))
    num_all_data=len(list2) 
    print("num_all_data:"+str(num_all_data))
    index_list=list(range(num_all_data))
    random.shuffle(index_list)
    trainDir=os.path.join(r"DataSource\OMDRdatasets\train",str(i))  # put training sets under this folder
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    testDir=os.path.join(r"DataSource\OMDRdatasets\test",str(i))   # put test sets under this folder
    if not os.path.exists(testDir):
        os.mkdir(testDir)
    num=0
    for j in index_list:
        file_name=os.path.join(os.path.join(dir,str(i)),list2[j])
        if num<num_all_data*0.9*data_capacity:
            copy2(file_name,trainDir)
        elif num<=num_all_data*data_capacity:
            copy2(file_name,testDir)
        num+=1
