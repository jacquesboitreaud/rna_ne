# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:27:35 2020

@author: jacqu

Parsing csv with 3D motifs occurences 
"""

import pickle
import pandas as pd 
import numpy as np

df = pd.read_csv('rna_3dmotifs.csv', header =None)
cpt= 0 # motifs counter 

d = {}

for i, row in df.iterrows():
    
    # Check if it is a new motif : starts with > 
    if(row[0].startswith('>')):
        m_name = row[0]
        print(m_name)
        cpt +=1
        motif = m_name
        
    else:
        
        positions = []
        
        for j in range(row.shape[0]):
            v = row[j]
            if(type(v)==str):
                
                try:
                    pdb, mdl, chain, nt, pos = v.split('|')
                except:
                    print(v)
                    try:
                        v, _ =v.split('|||')
                    except : 
                        v, _ = v.split('||')[:2]
                    pdb, mdl, chain, nt, pos = v.split('|')
                    
                pdb = pdb.lower()
                positions.append(pos)
                
                print(f'pdb {pdb}, mdl {mdl}, chain {chain}, nt {nt}, pos {pos}')
                
        # Add pdb, chain, positions to dict 
        d[pdb]= (motif, chain, positions)
        
print(d)

with open('3dmotifs_dict.pickle', 'wb') as f:
    pickle.dump(d,f)
        
        