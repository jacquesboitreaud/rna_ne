# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:03:01 2020

@author: jacqu

Computes pretrained embeddings for nucleotides, 
adds as a node feature and saves annotated nx graphs to 'args.savedir' 

"""

import sys, os
import pickle
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

import pandas as pd 
import numpy as np

import networkx as nx 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

if (__name__ == "__main__"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'data_processing'))
    
    from model import Model
    from data_processing.rnaDataset import rnaDataset, Loader
    from data_processing.rna_classes import *
    from utils import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--train_dir', help="path to training dataframe", type=str, default='data/chunks')
    parser.add_argument("--cutoff", help="Max number of train samples. Set to -1 for all in dir", 
                        type=int, default=100)
    parser.add_argument('--load_model_path', type=str, default = 'saved_model_w/model0_iter_7000.pth')
    # Where to save graphs with embeddings
    parser.add_argument('-o', '--savedir', type=str, default = 'data/with_embeddings')
    parser.add_argument('--batch_size', type=int, default = 16)
    
    ###########

    
    args=parser.parse_args()

    # config
    feats_dim, h_size, out_size=12, 16, 32 # dims 
    r1 = 1
    r2= 2
    
    #Loaders
    loaders = Loader(path=args.train_dir ,
                     attributes = ['angles', 'identity'],
                     N_graphs=args.cutoff, 
                     emb_size= feats_dim, 
                     true_edges=True, # Add the true edge types as an edge feature for validation & clustering
                     num_workers=0, 
                     batch_size=args.batch_size)
    
    with open('data/true_edge_map.pickle','wb') as f:
        pickle.dump(loaders.dataset.true_edge_map, f)
    
    N_edge_types = loaders.num_edge_types
    loader, _,_ = loaders.get_data()
    
    #Model & hparams
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    parallel=False
    
    # Model instance contains GNN and context GNN 
    model = Model(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, radii_params=(1,r1,r2), num_bases=-1).to(device).float()

    model.load_state_dict(torch.load(args.load_model_path))

    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    
    # Pass graphs to model and get node embeddings 
    model.eval()
    cpt=0
    with torch.no_grad():
        for batch_idx, (graph, pdb_ids) in enumerate(loader):

            graph=send_graph_to_device(graph,device)

            # Forward pass 
            model.GNN(graph) 
            
            #Unbatch and get node embeddings 
            graphs = dgl.unbatch(graph)
            
            for i,g in enumerate(graphs):
                nx_g= g.to_networkx(node_attrs=['h'], edge_attrs=['true_ET'])
                nx_g=nx.to_undirected(nx_g)
                
                with open(os.path.join(args.savedir,pdb_ids[i]+'.pickle'),'wb') as f:
                    pickle.dump(nx_g, f)
                cpt +=1 
    print(f'Saved {cpt} graphs with embeddings in ~/{args.savedir}')
            
            

