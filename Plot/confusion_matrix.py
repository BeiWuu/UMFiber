import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import matplotlib
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
from matplotlib.font_manager import FontProperties
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15

__author__='Research group of Huanyang Chen'

'''
Plot confusion matrix for VGGNet at 10 epochs
'''

confusion = np.array(([10,0,0,0,0,0,0,0,0,0],[0,10,0,0,0,0,0,0,0,0],[0,0,10,0,0,0,0,0,0,0],[0,0,0,10,0,0,0,0,0,0],[0,0,0,0,10,0,0,0,0,0],[0,0,0,0,0,9,0,0,1,0],[0,0,0,0,0,0,10,0,0,0],[0,0,0,0,0,0,0,9,0,1],[0,0,0,0,0,0,0,0,10,0],[0,0,0,0,0,0,0,0,0,10]))
plt.figure(figsize=(8,7))
plt.rcParams['font.size'] = 15
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.yticks(indices, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')

# Display specific data.
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        if first_index==second_index:
            plt.text(first_index, second_index, confusion[second_index][first_index],color='w',ha='center',va='center',fontsize=15)
        else:
            plt.text(first_index, second_index, confusion[second_index][first_index],ha='center',va='center',fontsize=15)
plt.savefig('conf_matrix.png', dpi=800)
plt.show()
