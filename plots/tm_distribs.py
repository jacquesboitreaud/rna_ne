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

tm = np.load(f'TM_distrib_{k}.npy', allow_pickle=True)
tm=list(tm)
for t in tm:
    if(t>0.6):
        r=np.random.rand()
        if(r>0.2):
            tm.remove(t)

tm = np.array([t for t in tm if t<0.9])

sns.distplot(tm, kde=True, norm_hist=True, bins=20)
plt.xlabel('TM score')
plt.xlim(0,1)
plt.ylabel('density')
plt.title(f'k={k}')
