# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:50:14 2020

@author: jacqu

Load graphs with embeddings,
visualize node embeddings for occurences of 3D motifs : 
    
    The goal is to see if we can see isomorphic nodes in motif occurences by jut looking at their embeddings 
    
"""

import pickle
import networkx as nx 
import torch

import os, sys
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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
    # Dict to collect embeddings, per edge type 
    d= {l:[] for l in etypes.values()}
    cpt = 0 # nbr of nodes read 
    
    stackings = {'S33', 'S35', 'S53', 'S55'}
    b = {'B35', 'B53'}
    canonical = {'CWW'}
    union = {'S33', 'S35', 'S53', 'S55', 'B35', 'B53', 'CWW'}
    nc = {e for e in d.keys() if (e not in union)}
    
    m_dict = {}
    
    # Loading fitted pca 

    
    for gid in graphs:
        
        embeddings = torch.zeros(40,emb_size)
        
        with open(os.path.join(data_dir,gid), 'rb') as f:
            g = pickle.load(f)
            
        with open(os.path.join('../data/motifs_chunks',gid),'rb') as f : 
            _ = pickle.load(f)
            motif = pickle.load(f)
            
        index = 0
        for (n, data) in g.nodes(data=True):
            
            embeddings[index]=data['h']
            index+=1
            
        embeddings = embeddings[:index]
        
        tf = embeddings
        
        if(motif in m_dict):
            m_dict[motif].append(embeddings)
        else:
            m_dict[motif]=[]
            m_dict[motif].append(embeddings)
            
    print('Built dictionary with motifs occurences')
    
    for m in m_dict : 
        if(len(m_dict[m])>1):
            
            print(m)
            occurrences = m_dict[m]
            
            # fit pca to all motif occurences 
            t = torch.cat([l for l in occurrences])
            t= np.array(t)
            pca = PCA()
            pca.fit(t)
            
            plt.figure()
            colors = sns.color_palette()
            for i in range(min(len(occurrences),8)):
                tf = pca.transform(occurrences[i])
                sns.scatterplot(tf[:,0], tf[:,1], c=colors[i], label=f'{i}')
            plt.title(f'{m}')
            plt.show()
