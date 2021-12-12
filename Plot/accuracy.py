import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15

__author__='Research group of Huanyang Chen'

'''
Plot the accuracy for VGGNet, Shallow CNNs (both at 10 epochs), and logitistic Regression with data capacity.
'''

x=[0,100,200,300,400,500,600,700,800,900,1000]
a1=np.array([0,0.5454545617103577,0.809523821,0.838709652,0.902438998,0.946428597,0.888888896,0.865451396,0.921130955,0.940463364,0.986486494541168])*100
a2=np.array([0,0.5937,0.679245,0.75342,0.7849462,0.79646,0.8045112,0.8150289,0.8208092,0.8497409,0.87323])*100
a3=np.array([0,0.09090909361839294,0.142857149,0.096774191,0.121951222,0.232142851,0.111111112,0.078125,0.0703125,0.219288796,0.27217061072587967])*100
plt.figure(figsize=(6,5))
plt.plot(x,a1,label='VGGNet')
plt.plot(x,a3,label='Shallow CNNs')
plt.plot(x,a2,label='logitistic Regression')
plt.xlabel('Data capacity')
plt.ylabel('Accuracy (%)')
plt.legend(loc="center right", frameon=False)
plt.xlim(0,1000)
plt.ylim(0,100)
'''
 #设置数字标签**
for a,b in zip(x,a1):
    if a==0:
        pass
    else:
        plt.text(a, b+0.7, r'%.1f%%' % b, ha='center', va= 'bottom',fontsize=9)

for a,b in zip(x,a2):
    if a==0:
        pass
    else:
        plt.text(a, b+0.7, r'%.1f%%' % b, ha='center', va= 'bottom',fontsize=9)

for a,b in zip(x,a3):
    if a==0:
        pass
    else:
        plt.text(a, b+0.7, r'%.1f%%' % b, ha='center', va= 'bottom',fontsize=9)
'''
plt.savefig('accuracy.png', dpi=800)
plt.show()