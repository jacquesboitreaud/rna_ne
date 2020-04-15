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
    
    parser.add_argument('-i', '--train_dir', help="path to graphs dataframe", type=str, default='data/motifs_chunks')
    parser.add_argument("--cutoff", help="Max number of train samples. Set to -1 for all in dir", 
                        type=int, default=-1)
    
    parser.add_argument('--load_model_path', type=str, default = 'saved_model_w/model0_HR.pth')
    
    
    # Where to save graphs with embeddings
    parser.add_argument('-o', '--savedir', type=str, default = 'data/with_embeddings')
    parser.add_argument('--batch_size', type=int, default = 1)
    
    ###########

    
    args=parser.parse_args()

    # config
    feats_dim, h_size, out_size=12, 16, 32 # dims 
    r1 = 1
    r2= 2
    
    use_fr3d_edges = True
    
    #Loaders
    loaders = Loader(path=args.train_dir ,
                     attributes = ['angles', 'identity'],
                     N_graphs=None, 
                     emb_size= feats_dim, 
                     true_edges=use_fr3d_edges,
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
                  num_rels=N_edge_types, radii_params=(1,r1,r2), num_bases=10).to(device).float()

    model.load_state_dict(torch.load(args.load_model_path))

    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    
    # Pass graphs to model and get node embeddings 
    model.eval()
    counter =0
    with torch.no_grad():
        for batch_idx, (graph, pdb_ids) in enumerate(loader):
            print('batch n° ',batch_idx , '/', len(loader))
            
            graph=send_graph_to_device(graph,device)
            

            # Forward pass 
            model.GNN(graph) 
            
            #Unbatch and get node embeddings 
            graphs = dgl.unbatch(graph)
            
            for i,g in enumerate(graphs):
                gid = pdb_ids[i]
                
                # Load networkx original graph 
                with open(f'{args.train_dir}/{gid}.pickle', 'rb') as f:
                    fr3d_g = pickle.load(f)
                    
                node_map = {nid : old_id for nid,old_id in enumerate(sorted(fr3d_g.nodes()))}
                # node map [ int label ] = label ('chain', 'pos')
                
                # A dict keyed by node ids (fr3d) with the embeddings 
                embeddings_attr = {node_map[k]: g.ndata['h'][int(k)] for k in range(len(g.nodes()))}
                # link nx_g node attr 'h' to true node labels (values in node_map )
                
                nx.set_node_attributes(fr3d_g, name='h', values= embeddings_attr)
                
                # Save back our graph with embeddings 
                with open(os.path.join(args.savedir,gid+'.pickle'),'wb') as f:
                    pickle.dump(fr3d_g, f)
                counter +=1
            print(counter, ' graphs embedded')
             
    
    print(f'Saved {counter} graphs with embeddings in ~/{args.savedir}')
            
            

