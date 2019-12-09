# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:10:02 2019

@author: jacqu
"""

import os 
import pickle


path = '/home/mcb/users/jboitr/data/DeepFRED_data'

for f in os.listdir(path):
  with open(os.path.join(path,f),'rb') as f:
        chunk = pickle.load(f)
        n_a, n_b, tmscore= pickle.load(f)
        if(tmscore<1):
            print(tmscore)