# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019
@author: jacqu


Magnesium binding with pretrained embeddings 

Investigate learned edges representations in context pred RGCN :
    
    for all pairs of edgetypes (r1, r2), compute pairwise distances between W_r1 and W_r2. 
    
    The more similar the weights, the more these edgetypes appear in similar contexts.

"""

import argparse
import sys, os 
import torch
import numpy as np

import pickle
import torch.utils.data
from torch import nn, optim

from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'tasks_processing'))
    sys.path.append(os.path.join(script_dir,'..'))
    
    from model import Model
    from data_processing.rna_classes import *
    from utils import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--pretrain_model_path', type=str, default = '../saved_model_w/model0_HR.pth',
                        help="path to rgcn to warm start embeddings")
    
    parser.add_argument('--edge_map', type=str, help='precomputed edge map for one-hot encoding. Set to None to rebuild. ', 
                        default = '../fr3d_edge_map.pickle')
    
     # =======

    args=parser.parse_args()
    
    # Choose model settings : 
    # 1 . 3 edgetypes (simplified)
    # 2 . 44 edgetypes (full)
    # 3 . 10 bases, 44 edgetypes (decomp)

    
    init_embeddings = Model(features_dim = 12, h_dim = 16, out_dim = 32, num_rels = 44, radii_params=(1,1,2),
                       num_bases = 10)
    init_embeddings.load_state_dict(torch.load(args.pretrain_model_path))
    print('Loaded context prediction RGCN')
    
    # Loading edge mapping : 
    with open(args.edge_map,'rb') as f:
        edge_map = pickle.load(f)
        edge_freqs = pickle.load(f)
    
    print(edge_map)
    
    print('Model params: ')
    for p in init_embeddings.GNN.layers[0].parameters():
        print(p.name, p.shape)
    
    w_comp = init_embeddings.GNN.layers[0].w_comp
    w_comp=w_comp.detach().numpy()
    
    """
    e1 = '9BR'
    e2 = 'TSW'
    i1, i2 = edge_map[e1] , edge_map[e2]
    print(w_comp[i1])
    print(w_comp[i2])
    """
    
    d = pairwise_distances(w_comp, metric = 'cosine')
    
    # FUll distances matrix reordered
    d, labels = reordered(d, edge_map)
    # heatmap
    mask = np.zeros_like(d)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(d, xticklabels = labels, yticklabels = labels, vmin = 0, vmax=0.6, mask = mask )
    
    # Small distances matrix reordered 
    d,labels = select_NC(d,edge_map)
    mask = np.zeros_like(d)
    mask[np.triu_indices_from(mask)] = True
    # heatmap
    plt.figure()
    sns.heatmap(d, xticklabels = labels, yticklabels = labels, vmin = 0, vmax=0.6, mask=mask)

    
    
        
        
        
        
        

    
        
