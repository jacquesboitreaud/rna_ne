# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Loads training set of edge pairs.
Instantiates gcn model 

Trains model on node pairs, under simLoss

## TODO : 
1/ Check the data computation pipeline (nodepairs, similarity)
2/ Think of loss function

- Loss computation : batchwise or summation of nodepairs losses for each graph in batch ? 
    
Meeting : 
    Think of similarity measures other than neighbor nucleotides RMSD 

"""
import sys
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rgcn import Model, simLoss
    from rnaDataset import rnaDataset, Loader
    from utils import *

    # config
    N=1 # num node features 
    N_types=44
    n_hidden = 16 # number of hidden units
    n_bases = -1 # use number of relations as number of bases
    n_hidden_layers = 1 # use 1 input layer, 1 output layer, no hidden layer
    n_epochs = 2 # epochs to train
    batch_size = 4
    
    #Load train set and test set
    loaders = Loader(num_workers=1, batch_size=batch_size)
    train_loader, _, test_loader = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False

    model = Model(num_nodes=N, h_dim=16, out_dim=1, num_rels=N_types, num_bases=-1).to(device)
    
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
        for batch_idx, (graph, idces, targets) in enumerate(train_loader):
            
            # Data checks
            if(len(targets)!=batch_size):
                print(f'targets length problem, not equal to batch size: l is {len(targets)}')
                print(targets)
            if(len(idces)!=batch_size):
                print(f'idces length problem, not equal to batch size: l is {len(idces)}')
                print(idces)
            # Embedding for each node
            graph=send_graph_to_device(graph,device)
            out = model(graph)
            
            #Compute loss term for each elemt in batch
            t_loss=0
            for i in range(batch_size): # for each elem in batch
                #TODO
                t_loss += (out[idces[i][0]]*out[idces[i][1]] - targets[i])**2 # (h1*h2 - r)^2
            # backward loss 
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
            for batch_idx, (graph,n1,n2,r) in enumerate(train_loader):
                out=model(graph)
                t_loss=0
                for i in range(batch_size): # for each elem in batch
                    t_loss += (out[idces[i][0]]*out[idces[i][1]] - targets[i])**2
                
            print(f'Validation loss at epoch {epoch}: {t_loss}')
        