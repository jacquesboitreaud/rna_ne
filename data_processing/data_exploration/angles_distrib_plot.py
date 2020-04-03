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
import pandas as pd

with open('angles_distrib_HR.pickle', 'rb') as f :
    d = pickle.load(f)
    
df = pd.DataFrame.from_dict(d)


plt.figure()
sns.distplot(d['alpha'], label='alpha torsion')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['beta'], label='beta torsion')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['gamma'], label='gamma torsion')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['epsilon'], label='epsilon torsion')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['zeta'], label='zeta torsion')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['delta'], label='delta torsion')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['chi'], label='chi torsion')
plt.plot([-1.8,-1.8],[0,2], c='r', linestyle ='dashed')
plt.plot([1.8,1.8],[0,2], c='r', linestyle ='dashed')
plt.xlabel('angle (rad)')
plt.legend()

plt.figure()
sns.distplot(d['psi'], label='psi angle')
plt.xlabel('angle (rad)')
plt.legend()

# Psi grouped by nucleotide : 

dfA = df[df['nt']=='A']
dfC = df[df['nt']=='C']
dfU = df[df['nt']=='U']
dfG = df[df['nt']=='G']

plt.figure()
sns.distplot(dfA['psi'], label='psi angle A')
sns.distplot(dfU['psi'], label='psi angle U')
sns.distplot(dfG['psi'], label='psi angle G')
sns.distplot(dfC['psi'], label='psi angle C')
plt.title('Base inclination angle by nucleotide type')
plt.xlabel('angle (rad)')
plt.legend()

