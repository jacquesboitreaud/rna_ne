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

    parser.add_argument('--train_dir', help="path to training dataframe", type=str, default='data/chunks_HR')
    parser.add_argument("--cutoff", help="Max number of train graphs. Set to -1 for all in dir", 
                        type=int, default=-1)
    parser.add_argument('--high_res', action='store_true', default=True) # train on 400 high res structures 
    
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

    parser.add_argument('--lr', type=float, default=1e-3) # Initial learning rate
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
    
    # Train directory and nodes : high resolution structures only (400)
    if(not args.high_res):
        train_nodes = pickle.load(open('data_processing/train_nodes.pickle','rb'))
    else:
        hr_structures = os.listdir('data/chunks_HR')
        print(f'>>> Pretraining on {len(hr_structures)} pdb structures')
        
    N=0
        
    for s in hr_structures : 
        with open('data/chunks_HR/'+s, 'rb') as f :
            g = pickle.load(f)
        N+= g.number_of_nodes()
        
    print(N)
            
    