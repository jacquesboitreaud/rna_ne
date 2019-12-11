# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:10:02 2019

@author: jacqu
"""

import os 
import pickle
import numpy as np



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
    from rgcn import Model, Loss
    from rnaDataset import rnaDataset, Loader
    from utils import *
    
    # Dict to get the edge embeddings 
    distances = {'target':[]}
    
    data_dir = '/home/mcb/users/jboitr/data/DF2'
    #data_dir = 'C:/Users/jacqu/Documents/GitHub/data/DeepFRED_data'
    
    #Load train set and test set
    loaders = Loader(path= data_dir,
                     N_graphs=cutoff, emb_size= 2, 
                     num_workers=8, batch_size=batch_size,EVAL=True)
    N_edge_types = loaders.num_edge_types
    train_loader, _, test_loader = loaders.get_data()
    with torch.no_grad():
        
        # get some from training set 
        for batch_idx, (graph, edges, tmscores,labels) in enumerate(train_loader):
            #print(labels)
            if(batch_idx%10==0):
                print('train batch ',batch_idx)
            
            tmscores=list(tmscores.numpy())
            distances['target']+=tmscores

        # get some from test set 
        for batch_idx, (graph, edges, tmscores,labels) in enumerate(test_loader):
            #print(labels)
            if(batch_idx%10==0):
                print('test batch ',batch_idx)
            
            tmscores=list(tmscores.numpy())
            distances['target']+=tmscores
                
        df = pd.DataFrame.from_dict(distances)
        print(df.shape)
        df.to_csv('tmscores.csv')