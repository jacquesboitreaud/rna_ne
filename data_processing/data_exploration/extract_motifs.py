# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:37:57 2020

@author: jacqu

Extract motifs graphs from RNA graphs and compute angles 

"""

import pickle 
import networkx as nx 
import os 


graphs_dir = '../../data/chunks'

with open('3dmotifs_dict.pickle', 'rb') as f :
    d = pickle.load(f)
    
for pdb, v in d.items():
    
    # Open pdb 
    try:
        g = pickle.load(open(os.path.join(graphs_dir,pdb+'.pickle'), 'rb'))
        
    except:
        #print(pdb , ' not found')
        pass