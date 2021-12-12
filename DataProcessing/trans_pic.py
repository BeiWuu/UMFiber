import os 
import numpy as np
import matplotlib.pyplot as plt 

__author__='Research group of Huanyang Chen'

'''
Convert normalized power data to pictures
'''

data_dir=r"DataSource\power_intensity"
pic_dir=r"DataSource\picFile" 
list=os.listdir(data_dir)
for i in list:
    list2=os.listdir(os.path.join(data_dir,str(i)))
    savefile1=os.path.join(pic_dir,str(i))
    if not os.path.isdir(savefile1):
        os.makedirs(savefile1)
    s=0
    for j in list2:
        plt.figure(figsize=(4,6))
        readfile=os.path.join(os.path.join(data_dir,str(i)),str(j))
        file=open(readfile,'r')
        file=np.loadtxt(file,delimiter='\t')
        column0,column1,column2,column3=file[0],file[1],file[2],file[3]
        # data normalization
        column0=(column0-column0[0])/0.41575+0.5177871316897175
        column1=(column1-column1[0])/1.57096+0.5177871316897175
        column2=(column2-column2[0])/0.33555+0.5177871316897175
        column3=(column3-column3[0])/0.36777000000000004+0.5177871316897175
        plt.plot(column1,'g',linewidth=1.5)    # Index finger
        plt.plot(column2+1,'m',linewidth=1.5)  # Middle finger
        plt.plot(column3+2,'c',linewidth=1.5)  # Ring finger
        plt.plot(column0+3,'r',linewidth=1.5)  # Little finger
        plt.xlim(0,2000)
        plt.ylim(0,4)
        plt.xticks([]) 
        plt.yticks([])
        #plt.show()
        plt.savefig(savefile1+'\\'+str(s)+'.png')
        plt.close('all')
        s+=1
