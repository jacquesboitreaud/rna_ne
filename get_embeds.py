# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:42:38 2019

@author: jacqu

Embed non canonical edges with model, and visualize FR3D clusters. 
"""

import sys
import pickle
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

import pandas as pd 
import numpy as np

if (__name__ == "__main__"):
    sys.path.append("dataloading")
    from rgcn import Model, Loss, resi
    from rnaDataset import rnaDataset, Loader
    from utils import *
    
    # Dict to get the edge embeddings 
    edges_d = {'label':[],'z1':[], 'z2':[], 'split':[]}
    residuals = {'pred':[], 'true':[]}
    
    load_model=True

    # config

    batch_size = 128
    load_path= 'saved_model_w/model0.pth'
    data_dir = '/home/mcb/users/jboitr/data/DF2'
    #data_dir = 'C:/Users/jacqu/Documents/GitHub/data/DeepFRED_data'
    
    cutoff=None
    
    #Load train set and test set
    loaders = Loader(path= data_dir,
                     N_graphs=cutoff, emb_size= 2, 
                     num_workers=8, batch_size=batch_size,EVAL=True)
    N_edge_types = loaders.num_edge_types
    train_loader, _, test_loader = loaders.get_data()
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats_dim, h_size, out_size=2, 8, 4 # dims 
    model = Model(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, num_bases=-1, num_hidden_layers=2).to(device)
    model.load_state_dict(torch.load(load_path))
    
    model.eval()
    t_loss=0
    with torch.no_grad():
        
        # get some from test set 
        for batch_idx, (graph, edges, tmscores,labels) in enumerate(test_loader):
            #print(labels)
            if(batch_idx%10==0):
                print('test batch ',batch_idx)
            
            n= len(labels) # batch size
            tmscores=tmscores.to(device)
            graph=send_graph_to_device(graph,device)
            z_e1, z_e2 = model(graph, edges)
            
            pred, true = resi(z_e1,z_e2,tmscores)
            
            
            # For loop over batch
            for i in range(n):
                # edge 1
                edges_d['label'].append(labels[i][0])
                edges_d['z1'].append(z_e1[i][0].item())
                edges_d['z2'].append(z_e1[i][1].item())
                edges_d['split'].append('test')
                # edge 2
                edges_d['label'].append(labels[i][1])
                edges_d['z1'].append(z_e2[i][0].item())
                edges_d['z2'].append(z_e2[i][1].item())
                edges_d['split'].append('test')
                
                residuals['pred'].append(pred[i].item())
                residuals['true'].append(true[i].item())
                
            df = pd.DataFrame.from_dict(edges_d)
            dfres = pd.DataFrame.from_dict(residuals)
            print(df.shape)
            df.to_csv('edge_embeddings0.csv')
            dfres.to_csv('resi0.csv')
