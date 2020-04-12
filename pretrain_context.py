# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019
@author: jacqu

Context prediction training on RNA graphs. 

!! Only preprocessed RNA graphs should be in 'args.train_dir' (using os.listdir to list graphs)

"""

import argparse
import sys, os 
import torch
import dgl

import pickle
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'data_processing'))
    
    from model import Model, pretrainLoss, draw_rec
    from data_processing.pretrainDataset import pretrainDataset, Loader
    from data_processing.rna_classes import *
    from utils import *
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', help="path to training dataframe", type=str, default='data/chunks')
    parser.add_argument("--cutoff", help="Max number of train graphs. Set to -1 for all in dir", 
                        type=int, default=-1)
    
    parser.add_argument('--save_path', type=str, default = 'saved_model_w/model0.pth')
    
    parser.add_argument('-p', '--num_processes', type=int, default=12) # Number of loader processes
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--fix_seed', action='store_true', default=False)
    parser.add_argument('--linear_transform', help='Learnable linear transform in loss', default=True)

    #Context prediction parameters 
    parser.add_argument('--K', type=int, default=1) # Number of hops of our GCN
    parser.add_argument('--r1', type=int, default=1) # Context ring inner radius
    parser.add_argument('--r2', type=int, default=2) # Context outer radius
    
    # Input graphs params: use simplified edges or use all edges  
    parser.add_argument('-e', '--edgetypes', type=str, default='all') # 'simplified or 'all'

    parser.add_argument('--lr', type=float, default=2e-3) # Initial learning rate
    parser.add_argument('--clip_norm', type=float, default=50.0) # Gradient clipping max norm

    parser.add_argument('--anneal_rate', type=float, default=0.9) # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=40000) # update learning rate every _ step
    parser.add_argument('--log_iter', type=int, default=100) # print loss metrics every _ step
    parser.add_argument('--save_iter', type=int, default=500) # save model weights every _ step

     # =======

    args=parser.parse_args()

    # config
    feats_dim, h_size, out_size=12, 16, 32 # dims 
    num_bases = 10 # nbr of bases for edges if 'all' edges used 
    simplified_edges = bool(args.edgetypes=='simplified')
    
    parallel = False
    
    # Train_dir 
    if(not args.debug):
        train_nodes = pickle.load(open('data_processing/train_nodes.pickle','rb'))
    else:
        train_nodes = pickle.load(open('data_processing/debug_nodes.pickle','rb'))
        

    
    #Loaders
    loaders = Loader(path = args.train_dir,
                    nodes_dict=train_nodes ,
                     simplified_edges=simplified_edges,
                     radii_params=(args.K,args.r1, args.r2),
                     attributes = ['angles', 'identity'],
                     N_graphs=args.cutoff, 
                     emb_size= feats_dim, 
                     num_workers=args.num_processes, 
                     batch_size=args.batch_size, 
                     fix_seed = args.fix_seed, 
                     debug = args.debug )
    
    # Tensorboard logging 
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    N_edge_types = loaders.num_edge_types
    train_loader, test_loader, _ = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    if(simplified_edges):
        b=-1
    else:
        b=num_bases 
        
    model = Model(features_dim=feats_dim, h_dim=h_size, out_dim=out_size, 
                  num_rels=N_edge_types, radii_params=(args.K,args.r1, args.r2), num_bases=b, dropout = 0.2).float()
    model.load_state_dict(torch.load('saved_model_w/model0_bases.pth'))

    #Print model summary
    print(model)
    model.to(device)
    map = ('cpu' if device == 'cpu' else None)

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print ("learning rate: %.6f" % scheduler.get_lr()[0])

    #Training loop
    model.train()
    total_steps=0

    for epoch in range(1, args.epochs+1):
        print(f'Starting epoch {epoch}')
        train_ep_loss, test_ep_loss = 0,0
        
        for batch_idx, (graph, ctx_graph, u_index, labels) in enumerate(train_loader):

            total_steps+=1 # count training steps
        
            graph=send_graph_to_device(graph,device)
            ctx_graph=send_graph_to_device(ctx_graph,device)
            labels = labels.to(device)

            # Forward pass
            model(graph, ctx_graph)
            
            # Get node embeddings 
            graphs = dgl.unbatch(graph)
            batch_size = len(graphs)
            
            h_v = torch.zeros((batch_size,out_size), device = device)
            for k in range(batch_size):
                h_v[k] = graphs[k].ndata['h'][u_index[k],:]
            
            if(parallel):
                h_v = model.module.linear_tf(h_v)
            else:
                h_v = model.linear_tf(h_v)
            
            # Get context embedding : average of anchor nodes             
            h_anchors = torch.zeros_like(h_v, device= device)
            ctx_graphs = dgl.unbatch(ctx_graph)
            for k in range(len(ctx_graphs)):
                is_anchor = [i for i,b in enumerate(list(ctx_graphs[k].ndata['anchor'])) if b>0]
                h = ctx_graphs[k].ndata['h']
                h_anchors[k] = torch.mean(h[is_anchor,:],dim=0)
                
            
            #Compute loss
            t_loss, dotprod = pretrainLoss(h_v, h_anchors, labels, v=False, show=bool(total_steps%args.log_iter==0 
                                                                                      and batch_size<128))
            optimizer.zero_grad()
            t_loss.backward()
            
            #Print & log
            per_item_loss = t_loss.item()/batch_size
            train_ep_loss += t_loss.item()
            if total_steps % args.log_iter == 0:
                figure = draw_rec(dotprod.view(-1,1), labels.view(-1,1))
                writer.add_figure('heatmap', figure, global_step=total_steps, close=True)
                writer.add_scalar('batchLoss/train', t_loss.item() , total_steps)
                print('epoch {}, opt. step n°{}, loss per it. {:.2f}'.format(epoch, total_steps, per_item_loss))
            
            del(t_loss)
            clip.clip_grad_norm_(model.parameters(),args.clip_norm)
            optimizer.step()

            # Annealing  LR
            if total_steps % args.anneal_iter == 0:
                 scheduler.step()
                 print ("learning rate: %.6f" % scheduler.get_lr()[0])
                 
            #Saving 
            if total_steps % args.save_iter == 0:
                torch.save( model.state_dict(), f"{args.save_path[:-4]}_iter_{total_steps}.pth")
                
        # Epoch logging 
        writer.add_scalar('epochLoss/train', train_ep_loss, epoch)
        print(f'Epoch {epoch}, total loss : {train_ep_loss}')
        
        
        
