# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:46:49 2019

@author: jacqu

TM distribs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

k=5

tm = np.load(f'TM_distrib_{k}.npy', allow_pickle=True)

sns.distplot(tm, kde=False, bins=20,norm_hist=False)
plt.xlim(0,1)
plt.xlabel('TM score')
plt.title(f'k={k}')
