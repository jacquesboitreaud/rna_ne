# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:51:06 2020

@author: jacqu

Plot roc curve for several models 
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt 

from sklearn.metrics import auc, roc_curve


fred_basic = '../tasks/fr3d_basic_mg.pickle'
fred_angles = '../tasks/fr3d_angles_mg.pickle'
warmstart_angles = '../tasks/warmstart_mg.pickle'

paths = [fred_basic, fred_angles, warmstart_angles]

for i,setting in enumerate(['basic', 'angles', 'pretrained']):
    
    setting_file = paths[i]
    with open(setting_file, 'rb') as f:
        truth = pickle.load(f)
        scores = pickle.load(f)

    # Compute roc 
    
    fpr, tpr, thresholds = roc_curve(y_true = truth, y_score = scores)
    roc_auc = auc(fpr, tpr)
    
    lw = 2
    plt.plot(fpr, tpr,
             lw=lw, label= setting +' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()