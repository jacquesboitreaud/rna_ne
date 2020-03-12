# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:00:34 2020

@author: jacqu

Reads parsed binding sites for PDBs
"""
import pickle


path = 'C:/Users/jacqu/Documents/MEGAsync Downloads/new_mg_res.p'

with open(path, 'rb') as f:
    
    mgdict = pickle.load(f)
    
"""
for k in mgdict.keys():
    print(k)
"""

samp = mgdict['3s1r.cif']
    
    
    

