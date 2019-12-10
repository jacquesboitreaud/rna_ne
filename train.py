# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Loads training set of edge pairs.
Instantiates + trains gcn model 

"""
import sys
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rgcn import Model, Loss
    from rnaDataset import rnaDataset, Loader
    from utils import *

    # config
    feats_dim, h_size, out_size=1, 16, 4 # dims 
    n_epochs = 2 # epochs to train
    batch_size = 3
    
    #Load train set and test set
    loaders = Loader(path= 'C:/Users/jacqu/Documents/GitHub/data/DeepFRED_data',
                     N_graphs=3, num_workers=0, batch_size=batch_size)
    N_edge_types = loaders.num_edge_types
    train_loader, _, test_loader = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False

    model = Model(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, num_bases=-1, num_hidden_layers=2).to(device)
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Start training using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    torch.manual_seed(1)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
    
    #Train & test
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        model.train()
        for batch_idx, (graph, edges, tmscores) in enumerate(train_loader):
            
            # Embedding for each node
            graph=send_graph_to_device(graph,device)
            out = model(graph)
            
            #Compute loss term for each elemt in batch
            t_loss = Loss(graph, edges, tmscores)
            optimizer.zero_grad()
            t_loss.backward()
            #clip.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            
            #logs and monitoring
            if batch_idx % 100 == 0:
                # log
                print('ep {}, batch {}, loss : {:.2f} '.format(epoch, 
                      batch_idx, t_loss.item()))
        
        # Validation pass
        model.eval()
        t_loss = 0
        with torch.no_grad():
            for batch_idx, (graph, edges, tmscore) in enumerate(test_loader):
                
                graph=send_graph_to_device(graph,device)
                out=model(graph)
                t_loss += Loss(graph, edges, tmscores)
                
            print(f'Validation loss at epoch {epoch}: {t_loss.item()}')
            
            # LOGS and SAVE : 
        