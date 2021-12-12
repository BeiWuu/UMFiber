import os 
import numpy as np

__author__='Research group of Huanyang Chen'

'''
Output the maximum and maximum power intensity of each finger
'''

dir=r"DataSource\power_intensity"  # The data collected by data acquisition card is stored in this folder.
list=os.listdir(dir)
intial0,intial1,intial2,intial3=1,1,1,1

# search for the max and max value
for i in list:
    list2=os.listdir(os.path.join(dir,str(i)))
    s=0
    for j in list2:
        readfile=os.path.join(dir,str(i),str(j))
        file=open(readfile,'r')
        file=np.loadtxt(file,delimiter='\t')
        column0,column1,column2,column3=file[0],file[1],file[2],file[3]  # power intensity of four fingers
        column0=(column0-column0[0])/0.41575+1.4089434176333533
        column1=(column0-column1[0])/0.83609+1.4089434176333533
        column2=(column0-column2[0])/0.47556+1.4089434176333533
        column3=(column0-column3[0])/0.54754+1.4089434176333533
        # data normalization
        max0=max(column0)
        if max0>intial0:
            intial0=max0
        max1=max(column1)
        if max1>intial1:
            intial1=max1
        max2=max(column2)
        if max2>intial2:
            intial2=max2
        max3=max(column3)
        if max3>intial3:
            intial3=max3
print(intial0,intial1,intial2,intial3)
