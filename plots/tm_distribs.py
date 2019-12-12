# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:46:49 2019

@author: jacqu

TM distribs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df=pd.read_csv('tmscores.csv')

targets = [t.strip('[') for t in df['target']]
targets = [t.strip(']') for t in targets]
targets.append('0.004')
targets = np.array([float(t) for t in targets])

targets=1/(targets)

sns.distplot(targets, kde=True, norm_hist=True, bins=20)
plt.xlabel('1/TMscore')
plt.ylabel('density')
