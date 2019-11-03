# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Reads RNA graphs annotated with FR3D edge labels 
"""

import numpy as np
import pickle 
import os 

DEBUG=True
gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"

cpt=0
for pdb_id in os.listdir(gr_dir):
    # pickle 'pdb_id'.pickle
    if(DEBUG):
        if(cpt==0):
            cpt +=1
            g = np.loadtxt(os.path.join(gr_dir,pdb_id))