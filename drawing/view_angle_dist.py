# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:19:55 2020

@author: jacqu

View angle distributions 
        
"""

import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 

import pickle

with open('C:/Users/jacqu/Documents/angles_distrib.pickle', 'rb') as f :
    d = pickle.load(f)


sns.distplot(d['chi'], label='chi torsion')
plt.legend()

plt.figure()
sns.distplot(d['psi'], label='psi angle')
plt.legend()

plt.figure()
sns.distplot(d['delta'], label='delta torsion')
plt.legend()