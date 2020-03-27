# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:58:18 2020

@author: jacqu
"""

import pickle 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open('../pdb_parsing_dict.pickle', 'rb') as f :
    d = pickle.load(f)
    
print(len(d), ' graphs parsed')
    
cpt=0
ok = []
num_perfect = []
bad_graphs = []
    
for gid, stats in d.items():
    cpt+=1
    print(gid)
    print(stats)
    
    try:
        frac_perfect = stats['perfect']/stats['num_nodes_init']
        num_perfect.append(stats['perfect'])
        ok.append(frac_perfect)
    except:
        frac_perfect = 0
        ok.append(frac_perfect)
        num_perfect.append(0)
    if(frac_perfect<0.5):
        bad_graphs.append(gid)
    
sns.distplot(num_perfect, norm_hist = False, kde=False, bins = 20)

tot_nodes = np.sum([di['num_nodes_init'] for di in d.values()])
# =========================================
print('***************************************************')
print(f' Total nodes in dataset : {tot_nodes}')
print(f' Nodes for which all angles were succesfully computed : {np.sum(num_perfect)}')

ok=np.array(ok)
bads = np.where(ok<0.5)[0]

print('Nbr of graphs with less than 50% nodes ok for all angles : ', len(bads))

with open('bad_graphs.pickle','wb') as f:
    pickle.dump(bad_graphs, f)



