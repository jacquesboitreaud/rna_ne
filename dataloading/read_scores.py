# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:10:02 2019

@author: jacqu
"""

import os 
import pickle
import numpy as np


path = '/home/mcb/users/jboitr/data/DF2'
tms=[]
cpt=0
for f in os.listdir(path):
  with open(os.path.join(path,f),'rb') as f:
        chunk,n_a, n_b, tmscore = pickle.load(f)
        if(cpt%100==0):
            print(cpt)
        tms.append(tmscore)
        cpt+=1
        
np.save('tmdist.npy',tms)