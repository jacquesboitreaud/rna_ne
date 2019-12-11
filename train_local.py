# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Loads training set of edge pairs.
Instantiates + trains gcn model 

"""
import sys
import random
import pickle
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
    
    load_model=False
    data_dir = 'C:/Users/jacqu/Documents/GitHub/data/DeepFRED_data'
    #data_dir = '/home/mcb/users/jboitr/data/DeepFRED_data'

    # config
    feats_dim, h_size, out_size=2, 8, 4 # dims 
    n_epochs = 10 # epochs to train
    batch_size = 3
    cutoff =None

    save_path, load_path = 'saved_model_w/model1.pth', 'saved_model_w/model1.pth'
    logs_path='saved_model_w/logs1.pth'
    
    #Load train set and test set
    loaders = Loader(path=data_dir ,
                     N_graphs=cutoff, emb_size= feats_dim, 
                     num_workers=0, batch_size=batch_size)
    
    logs_dict={'train_loss':[],'val_loss':[]}
    N_edge_types = loaders.num_edge_types
    train_loader, test_loader, _ = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    model = Model(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, num_bases=-1, num_hidden_layers=2).to(device)
    if(load_model):
        model.load_state_dict(torch.load(load_path))
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
        t_loss=0
        for batch_idx, (graph, edges, tmscores) in enumerate(train_loader):
            
            # Embedding for each node
            graph=send_graph_to_device(graph,device)
            tmscores=tmscores.to(device)
            z_e1, z_e2 = model(graph, edges)
            
            #Compute loss term for each elemt in batch
            b_loss = Loss(z_e1,z_e2, tmscores)
            optimizer.zero_grad()
            b_loss.backward()
            #clip.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            t_loss+=b_loss.item()
            #logs and monitoring
            if batch_idx % 10 == 0:
                # log
                print('ep {}, batch {}, loss : {:.2f} '.format(epoch, 
                      batch_idx, b_loss.item()))
                
        # End of training pass : add log to logs dict
        logs_dict['train_loss'].append(t_loss)
        
        # Validation pass
        model.eval()
        t_loss = 0
        with torch.no_grad():
            for batch_idx, (graph, edges, tmscores) in enumerate(test_loader):
                
                graph=send_graph_to_device(graph,device)
                tmscores=tmscores.to(device)
                z_e1, z_e2 = model(graph, edges)
                t_loss += Loss(z_e1,z_e2, tmscores).item()
                
            print(f'Validation loss at epoch {epoch}: {t_loss}')
            logs_dict['val_loss'].append(t_loss)
            
            # LOGS and SAVE : 
            if(epoch%5==0):     
                torch.save( model.state_dict(), save_path)
                pickle.dump(logs_dict, open(logs_path,'wb'))
                print(f"model saved to {save_path}")
        