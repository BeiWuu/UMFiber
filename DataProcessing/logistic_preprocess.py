import os
import pandas as pd
import numpy as np

__author__='Research group of Huanyang Chen'

'''
The function of this file is to preprocess for logitistic regression.
The initial sample contains 2000 points for each finger, and we take only 10 points by means of equal-interval sampling.
Reduce data and make it easier to process.
'''

list1=r"DataSource\power_intensity"
dfall = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,"class"])
save_path=r'DataSource\pd_data\pd_file.xlsx'

for i in os.listdir(list1):
    list2=os.listdir(os.path.join(list1,str(i)))
    for j in list2:
        rang=np.arange(1,2000,200).tolist()    # want
        full=range(2000)
        rang1=[k for k in full if k not in rang]   # unwanted
        df=pd.read_excel(os.path.join(list1,str(i),j),header=None)
        df=df.drop(columns=rang1)   # Filter out unwanted columns
        value=pd.DataFrame(df.values.reshape(1,40))
        value.insert(loc=40,column='class',value=[int(i)])   # Use i=0,1,...,9 to indicate the class of gesture
        dfall=pd.concat([dfall,value],axis=0)   # concat the dfall and value
        dfall.to_excel(save_path)

