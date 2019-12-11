# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:46:49 2019

@author: jacqu

TM distribs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

k=4

tm = np.load(f'../dataloading/final_dist.npy', allow_pickle=True)
tm=list(tm)

sns.distplot(tm, kde=True, norm_hist=True, bins=20)
plt.xlabel('TM score')
plt.xlim(0,1)
plt.ylabel('density')
plt.title(f'k={k}')
