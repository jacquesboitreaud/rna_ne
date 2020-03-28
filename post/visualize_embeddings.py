# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:50:14 2020

@author: jacqu

Load graphs with embeddings,
visualize node embeddings for different basepairs types
"""

import pickle
import networkx as nx 
import torch

import os, sys
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append('../data_processing')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--graphs_dir', help="path to training dataframe", type=str, default='../data/with_embeddings')
    
    ############
    
    args=parser.parse_args()
    
    # Params 
    data_dir = args.graphs_dir
    graphs = os.listdir(data_dir)
    emb_size = 32
    
    with open('../data/true_edge_map.pickle','rb') as f:
        edgemap = pickle.load(f)
    
    etypes = {l:t for (t,l) in loaders.dataset.true_edge_map.items()}
    cpt = 0 # nbr of nodes read 
    
    embeddings = torch.zeros(10000,emb_size)
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
    stackings = {'S33', 'S35', 'S53', 'S55'}
    b = {'B35', 'B53'}
    canonical = {'CWW'}
    union = {'S33', 'S35', 'S53', 'S55', 'B35', 'B53', 'CWW'}
    nc = {e for e in d.keys() if (e not in union)}
    
    for etype, indexes in d.items():
        if(etype in stackings):
            subtype = x2d[indexes,:]
            sns.scatterplot(subtype[:,0], subtype[:,1], color='g')
        if(etype in canonical):
            subtype = x2d[indexes,:]
            sns.scatterplot(subtype[:,0], subtype[:,1], color='b')
        if(etype in nc):
            subtype = x2d[indexes,:]
            sns.scatterplot(subtype[:,0], subtype[:,1], color='r')
