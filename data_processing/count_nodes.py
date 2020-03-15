# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:04:24 2020

@author: jacqu
"""

import pickle
import os

gr_dir = '../data/chunks'
"""
graphs = os.listdir(gr_dir)

cpt=0 
graphs_cpt = 0
d = {}

for gid in graphs:
    graphs_cpt+=1

    with open(os.path.join(gr_dir,gid),'rb') as f:
        g = pickle.load(f)
        nn = g.number_of_nodes()
        cpt += nn
    d[gid]=nn
    if(nn<6)
    
with open('nodecounts.pickle', 'wb') as f:
    pickle.dump(d,f)
print(cpt, ' nodes')
print(graphs_cpt, ' graphs')
"""

with open('nodecounts.pickle', 'rb') as f:
    d= pickle.load(f)
print(len(d))