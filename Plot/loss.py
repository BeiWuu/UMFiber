import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15

__author__='Research group of Huanyang Chen'

'''
Plot loss and accurcy for VGGNet changing with epochs
'''

x=[0,1,2,3,4,5,6,7,8,9]
TrainLoss=[2.9335557392665317,1.1048650017806463,0.44216389687997953,0.38057950884103775,0.24610628081219538,0.13857177724795683,0.10203411616384983,0.1622826253463115,0.12984625596020902,0.07346550069217171]
ValLoss=[1.7528492212295532,0.5549745559692383,0.337452694773674,0.09839173965156078,0.1598808318376541,0.03731224709190428,0.018392146797850728,0.042196910828351974,0.1350468061864376,0.0972907803952694]
TrainAcc=[0.21316964285714285,0.6191716279302325,0.8504464285714286,0.8833085298538208,0.9346478155681065,0.9418402782508305,0.9663938496794019,0.9507688496794019,0.9474206353936877,0.9737103155681065]
ValAcc=[0.45967741310596466,0.7620967626571655,0.8807963728904724,0.9921875,0.9203628897666931,0.984375,1.0,0.9921875,0.9526209533214569,0.9838709533214569]
TrainAcc=[100*i for i in TrainAcc]
ValAcc=[100*i for i in ValAcc]
fig,ax1=plt.subplots(figsize=(5,4))
ax1.plot(x,TrainLoss,'r--',label='Training loss')
ax1.plot(x,ValLoss,'r',label='Validation loss')
ax1.set_ylim(0,)
ax1.set_ylabel('Loss')

ax2=ax1.twinx()
ax2.plot(x,TrainAcc,'g--',label='Training acc.')
ax2.plot(x,ValAcc,'g',label='Validation acc.')
ax2.set_ylim(0,)
ax2.set_ylabel('Accuracy (%)')
fig.legend(loc="center right", bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes,frameon=False)
plt.xlim(0,9)
plt.xlabel('Epoch')
'''
# Set label
for a,b in zip(x,TrainAcc):
    plt.text(a, b-0.011, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)
'''
plt.savefig('loss.png', dpi=800)
plt.show()