# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:27:35 2020

@author: jacqu

Parsing csv with 3D motifs occurences 
"""

import pickle
import pandas as pd 
import numpy as np


with open('3dmotifs_dict.pickle', 'rb') as f:
    d= pickle.load(f)
        
    
ranked_d = {}

refs = {}

for pdb,v in d.items():
    
    print(pdb)
    l = len(v)
    
    for i in range(l):
        
        # for each motif found in pdb 
        
        motif, chain, nts, r = v[i]
        
        ranked_d[(pdb,motif)]=r
        
        if(i==0):
            refs[motif]=pdb
        
with open('discrepancy_ranks_dict.pickle', 'wb') as f:
    pickle.dump(ranked_d, f)
    pickle.dump(refs, f)