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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'tasks_processing'))
    sys.path.append(os.path.join(script_dir,'..'))
    
    from model import RGCN, classifLoss, draw_rec
    from tasks_processing.mgDataset import mgDataset, Loader
    from data_processing.rna_classes import *
    from utils import *
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', help="path to training dataframe", type=str, default='data/mg_graphs')
    parser.add_argument("--cutoff", help="Max number of train samples. Set to -1 for all in dir", 
                        type=int, default=-1)
    parser.add_argument("-f","--fr3d", action='store_true', help="Set to true to use original FR3D graphs (baseline)",
                        default=True)
    
    parser.add_argument('--save_path', type=str, default = 'saved_model_w/model0.pth')
    parser.add_argument('--load_model', type=bool, default=False)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    
    parser.add_argument('-p', '--num_processes', type=int, default=4) # Number of loader processes
    
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--layers', type=int, default=2) # nbr of layers in RGCN 

    parser.add_argument('--lr', type=float, default=1e-3) # Initial learning rate
    parser.add_argument('--clip_norm', type=float, default=50.0) # Gradient clipping max norm
    parser.add_argument('--anneal_rate', type=float, default=0.9) # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=1000) # update learning rate every _ step
    
    parser.add_argument('--log_iter', type=int, default=5) # print loss metrics every _ step
    parser.add_argument('--save_iter', type=int, default=1000) # save model weights every _ step

     # =======

    args=parser.parse_args()

    # config
    feats_dim, h_size, out_size=6, 16, 1 # dims 
    
    #Loaders
    if(args.fr3d):
        print('********** Baseline model training, using FR3D graphs and edgetypes ******')
    else:
        print('********** Training model with learned nucleotide embeddings ***********')
    loaders = Loader(path=args.train_dir ,
                     true_edges=args.fr3d, # Whether we use true FR3D graphs or embeddings graphs 
                     attributes = ['A','U','G','C','chi','gly_base'],
                     N_graphs=args.cutoff, 
                     emb_size= feats_dim, 
                     num_workers=args.num_processes, 
                     batch_size=args.batch_size)
    
    # Tensorboard logging 
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    N_edge_types = loaders.num_edge_types
    train_loader, test_loader, _ = loaders.get_data()
    
    #Model & hparams
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    parallel=False
    
    # Simple RGCN instance for node classification 
    model = RGCN(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, num_layers = args.layers, num_bases=-1, pool=False).to(device).float()

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
        print(f'Starting epoch {epoch}')
        train_ep_loss, test_ep_loss = 0,0
        accuracy_num, accuracy_denom, pos_pred = 0,0, 0
        
        for batch_idx, (graph, pdbids) in enumerate(train_loader):

            total_steps+=1 # count training steps
            
            graph=send_graph_to_device(graph,device)

            # Forward pass
            model(graph)
            
            # Get node embeddings 
            h = graph.ndata['h'].view(-1,1)
            labels = graph.ndata['Mg_binding'].float().view(-1,1)
            #Compute loss
            t_loss = classifLoss(h, labels, show=False) #show=bool(total_steps%args.log_iter==0))
            optimizer.zero_grad()
            t_loss.backward()
            
            # Epoch accuracy 
            pred = torch.round(torch.sigmoid(h))
            pos_pred += torch.sum(pred)
            correct = (pred==labels).float().sum()
            accuracy_num += correct 
            accuracy_denom += pred.shape[0]
            
            #Print & log
            train_ep_loss += t_loss.item()
            if total_steps % args.log_iter == 0:
                figure = draw_rec(h.view(-1,1), labels.view(-1,1))
                writer.add_figure('heatmap', figure, global_step=total_steps, close=True)
                writer.add_scalar('batchLoss/train', t_loss.item()  , total_steps)
                print('epoch {}, opt. step n°{}, loss {:.2f}'.format(epoch, total_steps, t_loss.item()))
            
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
        
        accuracy = accuracy_num/accuracy_denom
        frac_pos_pred = pos_pred/accuracy_denom
        print(f'epoch {epoch}, train accuracy : {accuracy}, frac positive pred : {frac_pos_pred}')
        # Epoch logging 
        writer.add_scalar('epochLoss/train', train_ep_loss, epoch)
        writer.add_scalar('epochAcc/train', accuracy, epoch)
        
        # Validation pass
        model.eval()
        accuracy_num, accuracy_denom, pos_pred = 0,0, 0
        with torch.no_grad():
            for batch_idx, (graph, pdbids ) in enumerate(test_loader):

                graph=send_graph_to_device(graph,device)

                # Forward pass 
                model(graph) 
                
                h=graph.ndata['h'].view(-1,1)
                labels = graph.ndata['Mg_binding'].float().view(-1,1)
            
                #Compute loss
                t_loss = classifLoss(h, labels, show = False)
                test_ep_loss += t_loss.item()
                
                # Epoch accuracy 
                pred = torch.round(torch.sigmoid(h))
                pos_pred += torch.sum(pred)
                correct = (pred==labels).float().sum()
                accuracy_num += correct 
                accuracy_denom += pred.shape[0]
                
            accuracy = accuracy_num/accuracy_denom
            frac_pos_pred = pos_pred /accuracy_denom
            print(f'epoch {epoch}, test accuracy : {accuracy}, frac positive pred : {frac_pos_pred}')
                    
        # Epoch logging
        writer.add_scalar('epochLoss/test', test_ep_loss, epoch)
        writer.add_scalar('epochAcc/test', accuracy, epoch)
        
