# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:58:18 2020

@author: jacqu
"""

import pickle 

with open('pdb_parsing_dict.pickle', 'rb') as f :
    d = pickle.load(f)
    
cpt=0
    
for gid, stats in d.items():
    cpt+=1
    print(gid)
    print(stats)
    
    if(cpt>10):
        break

