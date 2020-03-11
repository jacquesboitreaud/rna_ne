# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:50:14 2020

@author: jacqu

Load graphs with embeddings,
visualize node embeddings for different basepairs types
"""

import pickle
import networkx as nx 

import os, sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

data_dir = '../data/with_embeddings'
graphs = os.listdir(data_dir)

etypes = {l:t for (t,l) in loaders.dataset.true_edge_map.items()}
cpt = 0 # nbr of nodes read 

embeddings = torch.zeros(10000,64)
# Dict to collect embeddings, per edge type 
d= {l:[] for l in etypes.values()}

for g in graphs:
    
    with open(os.path.join(data_dir,g), 'rb') as f:
        g = pickle.load(f)
        

    for (u,v, data) in g.edges(data=True):
        eid = data['true_ET'].item()
        etype = etypes[eid]
        
        u, v = g.nodes[u], g.nodes[v]
        h1, h2 = u['h'], v['h']
        
        d[etype].append(cpt)
        embeddings[cpt]=h1
        cpt+=1
        d[etype].append(cpt)
        embeddings[cpt]=h2
        
        
# PCA fit all embeddings 
        
embeddings = embeddings[:cpt+1,:]

# PCA 
pca = PCA(n_components=2)
x2d = pca.fit_transform(embeddings)     
    
# plot 
for etype, indexes in d.items():
    if(etype in ['CWW', 'CWS', 'CSW', 'CHS', 'CSH', 'CWH','CHW']):
        subtype = x2d[indexes,:]
        sns.scatterplot(subtype[:,0], subtype[:,1], label = etype)
        plt.legend()
