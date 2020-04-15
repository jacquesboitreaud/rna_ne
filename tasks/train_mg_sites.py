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

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc

import matplotlib.pyplot as plt

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'tasks_processing'))
    sys.path.append(os.path.join(script_dir,'..'))
    
    from model_mg_sites import RGCN
    from model import Model
    from tasks_processing.mg_sites_Dataset import mgDataset, Loader
    from data_processing.rna_classes import *
    from utils import *
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', help="path to training dataframe", type=str, default='data/mg_sites')
    parser.add_argument("--cutoff", help="Max number of train samples. Set to -1 for all in dir", 
                        type=int, default=None)
    
    parser.add_argument("-e","--embeddings", action='store_true', help="Initialize with pretrained embeddings.",
                        default=True)
    parser.add_argument('-m', '--pretrain_model_path', type=str, default = '../saved_model_w/model0_HR.pth',
                        help="path to rgcn to warm start embeddings")
    
    parser.add_argument('--load_model', type=bool, default=False)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('-p', '--num_processes', type=int, default=4) # Number of loader processes

    parser.add_argument('--layers', type=int, default=2) # nbr of layers in RGCN 
    parser.add_argument('--edge_map', type=str, help='precomputed edge map for one-hot encoding. Set to None to rebuild. ', 
                        default = 'mg_edge_map.pickle')

    parser.add_argument('--lr', type=float, default=6e-4) # Initial learning rate
    parser.add_argument('--clip_norm', type=float, default=50.0) # Gradient clipping max norm
    parser.add_argument('--anneal_rate', type=float, default=0.8) # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=400) # update learning rate every _ step
    
    parser.add_argument('--log_iter', type=int, default=50) # print loss metrics every _ step
    
     # =======

    args=parser.parse_args()

    # config
    attributes = ['angles','identity'] # node features to use
    
    if args.embeddings: # Initialize with pretrained embeddings 
        
        init_embeddings = Model(features_dim = 12, h_dim = 16, out_dim = 32, num_rels = 44, radii_params=(1,1,2),
                           num_bases =10)
        init_embeddings.load_state_dict(torch.load(args.pretrain_model_path))
        print('Loaded RGCN layer to warm-start embeddings')
        
        feats_dim, h_size, out_size=32, 16, 16 # dims 
    else:
        print('Baseline model training, using FR3D graphs and edgetypes')
        
        if( attributes == ['angles', 'identity']):
            feats_dim, h_size, out_size=12, 16, 2 # dims 
        elif 'angles' in attributes :
            feats_dim, h_size, out_size=8, 16, 2 # dims 
        else:
            feats_dim, h_size, out_size=4, 16, 2  # dims 
            
    # Define saved model name : 
    if args.embeddings : 
        name = 'warmstart_mg'
    elif 'angles' in attributes : 
        name = 'fr3d_angles_mg'
    else:
        name = 'fr3d_basic_mg'
    save_path = f'saved_model_w/{name}.pth'
            
    bases = 10 
    
    # Weighted loss 
    weights = torch.tensor([1.,1.])
    
    #Loaders
    loaders = Loader(path=args.train_dir ,
                     true_edges= True, # Whether we use true FR3D edgetypes or simplied edgetypes
                     attributes = attributes,
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
    weights = weights.to(device)
    
    # Simple RGCN instance for node classification 
    model = RGCN(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, num_layers = args.layers, num_bases=bases).to(device).float()
    if(args.embeddings):
        init_embeddings.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(save_path))
    
    m=nn.LogSoftmax(dim=1)
    criterion = nn.NLLLoss(weight = weights, reduction = 'sum')

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
    best_before = 1e5

    for epoch in range(1, args.epochs+1):
        print('*********************************')
        print(f'Starting epoch {epoch}')
        train_ep_loss, test_ep_loss = 0,0
        pos_pred = 0
        
        for batch_idx, (graph, pdbids, labels) in enumerate(train_loader):

            total_steps+=1 # count training steps
            
            graph=send_graph_to_device(graph,device)
            labels = torch.tensor(labels).to(device)
            
            # Warm start embeddings 
            if(args.embeddings):
                init_embeddings.GNN(graph)

            # Forward pass
            h = model(graph)
        
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
                writer.add_scalar('precision/train', p , total_steps)
                writer.add_scalar('recall/train', r  , total_steps)
                writer.add_scalar('f1/train', f1  , total_steps)
                print('epoch {}, opt. step n°{}, loss {:.2f}'.format(epoch, total_steps, t_loss.item()))
                print(f'precision: {p}, recall: {r}, f1-score: {f1}')
            
            del(t_loss)
            clip.clip_grad_norm_(model.parameters(),args.clip_norm)
            optimizer.step()

            # Annealing  LR
            if total_steps % args.anneal_iter == 0:
                 scheduler.step()
                 print ("learning rate: %.6f" % scheduler.get_lr()[0])
        
        print(f'epoch {epoch}, loss : {train_ep_loss}, N positive pred : {pos_pred}')
        # Epoch logging 
        writer.add_scalar('epochLoss/train', train_ep_loss, epoch)
        
        # Validation pass
        model.eval()
        pos_pred = 0
        
        test_true, test_pred, scores = [],[], []
        
        with torch.no_grad():
            for batch_idx, (graph, pdbids, labels ) in enumerate(test_loader):

                graph=send_graph_to_device(graph,device)
                labels = torch.tensor(labels).to(device)
                
                if(args.embeddings):
                    init_embeddings.GNN(graph)
                    
                # Forward pass 
                h= model(graph) 
            
                #Compute loss
                t_loss = criterion( m(h), labels)
                test_ep_loss += t_loss.item()
                
                # Epoch accuracy 
                _, pred = torch.max(m(h), dim=1)
                score_pos = m(h)[:,1].cpu().detach().numpy() # proba of label 1 
                pos_pred += torch.sum(pred).float()
                
                # Confusion matrix : 
                true, pred = labels.cpu().detach(), pred.cpu().detach()
                
                test_true.append(true)
                test_pred.append(pred)
                scores.append(score_pos)
                
                
            truth, preds = np.concatenate(test_true), np.concatenate(test_pred)
            scores = np.concatenate(scores)
            
            p = precision_score(truth, preds)   
            r = recall_score(truth, preds)   
            f1 = 2*(p*r)/(p+r)
            
            print('*************** Validation pass *********************')
            print(f'epoch {epoch}, Validation loss : {test_ep_loss}, N positive pred : {pos_pred}')
            print(f'precision: {p}, recall: {r}, f1-score: {f1}')
                    
        # Epoch logging
        writer.add_scalar('epochLoss/test', test_ep_loss, epoch)
        
        writer.add_scalar('precision/test', p , total_steps)
        writer.add_scalar('recall/test', r  , total_steps)
        writer.add_scalar('f1/test', f1  , total_steps)
        
        
        # If better test loss
        
        if test_ep_loss < best_before :
            best_before = test_ep_loss
            print('valid loss decreased. saving.')
            
            #Saving model 
            torch.save( model.state_dict(), save_path)
            
        with open(f'{name}.pickle', 'wb') as f : 
            pickle.dump(truth,f)
            pickle.dump(scores,f)
