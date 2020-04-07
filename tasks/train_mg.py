# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019
@author: jacqu


Magnesium binding with pretrained embeddings 

Trains simple RGCN instance for magnesium binding prediction 

"""

import argparse
import sys, os 
import torch
import numpy as np

import pickle
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, precision_score, recall_score

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'tasks_processing'))
    sys.path.append(os.path.join(script_dir,'..'))
    
    from model_mg import RGCN
    from tasks_processing.mgDataset import mgDataset, Loader
    from data_processing.rna_classes import *
    from utils import *
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', help="path to training dataframe", type=str, default='data/mg_annotated_graphs')
    parser.add_argument("--cutoff", help="Max number of train samples. Set to -1 for all in dir", 
                        type=int, default=300)
    parser.add_argument("-f","--fr3d", action='store_true', help="Set to true to use original FR3D graphs (baseline)",
                        default=True)
    
    parser.add_argument("-e","--embeddings", action='store_true', help="Use pretrained embeddings.",
                        default=True)
    
    parser.add_argument('--save_path', type=str, default = 'saved_model_w/model0.pth')
    parser.add_argument('--load_model', type=bool, default=False)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('-p', '--num_processes', type=int, default=4) # Number of loader processes

    parser.add_argument('--layers', type=int, default=2) # nbr of layers in RGCN 
    parser.add_argument('--edge_map', type=str, help='precomputed edge map for one-hot encoding. Set to None to rebuild. ', 
                        default = 'mg_edge_map.pickle')

    parser.add_argument('--lr', type=float, default=1e-3) # Initial learning rate
    parser.add_argument('--clip_norm', type=float, default=50.0) # Gradient clipping max norm
    parser.add_argument('--anneal_rate', type=float, default=0.9) # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=1000) # update learning rate every _ step
    
    parser.add_argument('--log_iter', type=int, default=5) # print loss metrics every _ step
    parser.add_argument('--save_iter', type=int, default=1000) # save model weights every _ step
    
    

     # =======

    args=parser.parse_args()

    # config
    if(args.embeddings):
        feats_dim, h_size, out_size=12, 16, 16 # dims 
    else:
        feats_dim, h_size, out_size=32, 16, 16 # dims 
    bases = 10 
    
    weights = torch.tensor([1.,100.])
    
    #Loaders
    if(args.fr3d):
        print('********** Baseline model training, using FR3D graphs and edgetypes ******')
    else:
        print('********** Training model with learned nucleotide embeddings ***********')
    loaders = Loader(path=args.train_dir ,
                     true_edges=args.fr3d, # Whether we use true FR3D graphs or embeddings graphs 
                     attributes = ['angles', 'identity'],
                     N_graphs=args.cutoff, 
                     emb_size= feats_dim, 
                     num_workers=args.num_processes, 
                     batch_size=args.batch_size, 
                     prebuilt_edge_map = args.edge_map )
    
    # Tensorboard logging 
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    N_edge_types = loaders.num_edge_types
    train_loader, test_loader, _ = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    
    # Simple RGCN instance for node classification 
    model = RGCN(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, num_layers = args.layers, num_bases=bases, pool=False).to(device).float()
    weights = weights.to(device)
    
    m=nn.LogSoftmax(dim=1)
    criterion = nn.NLLLoss(weight = weights, reduction = 'sum')
    

    if(args.load_model):
        model.load_state_dict(torch.load(args.load_path))

    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print ("learning rate: %.6f" % scheduler.get_lr()[0])

    #Train & test
    model.train()
    total_steps=0 # nbr of optim steps 

    for epoch in range(1, args.epochs+1):
        print('*********************************')
        print(f'Starting epoch {epoch}')
        train_ep_loss, test_ep_loss = 0,0
        pos_pred = 0
        
        for batch_idx, (graph, pdbids) in enumerate(train_loader):

            total_steps+=1 # count training steps
            
            graph=send_graph_to_device(graph,device)

            # Forward pass
            h = model(graph)
            #h = graph.ndata['h'].view(-1,out_size)
            
            # Get mg labels 
            labels = graph.ndata['Mg_binding'].long()
            
            #Compute loss
            t_loss = criterion( m(h), labels)
            optimizer.zero_grad()
            t_loss.backward()
            
            # Epoch accuracy 
            _, pred = torch.max(m(h), dim=1)
            pos_pred += torch.sum(pred).float()
            
            # Confusion matrix : 
            true, pred = labels.cpu().detach(), pred.cpu().detach()
            p = precision_score(true, pred)   
            r = recall_score(true, pred) 
            f1 = 2*(p*r)/(p+r)
            
            
            #Print & log
            train_ep_loss += t_loss.item()
            if total_steps % args.log_iter == 0:

                writer.add_scalar('batchLoss/train', t_loss.item()  , total_steps)
                print('epoch {}, opt. step n°{}, loss {:.2f}'.format(epoch, total_steps, t_loss.item()))
                print(f'precision: {p}, recall: {r}, f1-score: {f1}')
            
            del(t_loss)
            clip.clip_grad_norm_(model.parameters(),args.clip_norm)
            optimizer.step()

            # Annealing  LR
            if total_steps % args.anneal_iter == 0:
                 scheduler.step()
                 print ("learning rate: %.6f" % scheduler.get_lr()[0])
                 
            #Saving model 
            if total_steps % args.save_iter == 0:
                torch.save( model.state_dict(), f"{args.save_path[:-4]}_iter_{total_steps}.pth")
        
        print(f'epoch {epoch}, loss : {train_ep_loss}, N positive pred : {pos_pred}')
        # Epoch logging 
        writer.add_scalar('epochLoss/train', train_ep_loss, epoch)
        
        # Validation pass
        model.eval()
        pos_pred = 0
        with torch.no_grad():
            for batch_idx, (graph, pdbids ) in enumerate(test_loader):

                graph=send_graph_to_device(graph,device)

                # Forward pass 
                h= model(graph) 
                #h=graph.ndata['h'].view(-1,out_size)
                
                labels = graph.ndata['Mg_binding'].long()
            
                #Compute loss
                t_loss = criterion( m(h), labels)
                test_ep_loss += t_loss.item()
                
                # Epoch accuracy 
                _, pred = torch.max(m(h), dim=1)
                pos_pred += torch.sum(pred).float()
                
                # Confusion matrix : 
                true, pred = labels.cpu().detach(), pred.cpu().detach()
                p = precision_score(true, pred)   
                r = recall_score(true, pred)   
                f1 = 2*(p*r)/(p+r)
            
            print('*************** Validation pass *********************')
            print(f'epoch {epoch}, Validation loss : {test_ep_loss}, N positive pred : {pos_pred}')
            print(f'precision: {p}, recall: {r}, f1-score: {f1}')
                    
        # Epoch logging
        writer.add_scalar('epochLoss/test', test_ep_loss, epoch)
        
