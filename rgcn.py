# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

RGCN to learn node embeddings on RNA graphs , with edge types 

https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from functools import partial
import dgl
from dgl import mean_nodes

from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GATConv, RelGraphConv

class Model(nn.Module):
    # Computes embeddings for all nodes
    # No features
    def __init__(self, features_dim, h_dim, out_dim , num_rels, num_bases=-1, num_hidden_layers=2):
        super(Model, self).__init__()
        
        self.features_dim, self.h_dim, self.out_dim = features_dim, h_dim, out_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()
        
        #self.attn = GATConv(in_feats=self.out_dim, out_feats=self.out_dim,num_heads=1)
        self.dense = nn.Linear(self.out_dim,2)

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
            self.layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(self.h_dim, self.out_dim, self.num_rels, activation=nn.ReLU())
        self.layers.append(h2o)


    def forward(self, g):
        #print('edge data size ', g.edata['one_hot'].size())
        
        for layer in self.layers:
             #print(g.ndata['h'].size())
             #print(g.edata['one_hot'].size())
             g.ndata['h']=layer(g,g.ndata['h'],g.edata['one_hot'])
        
        g.ndata['h']=self.dense(g.ndata['h'])
        
def Loss(g, edges, tmscores):
    # Takes batches graph and labels, computes loss 
    bnn = g.batch_num_nodes
    N = len(bnn) # batch size 
    loss=0
    for i in range(N):
        u1 = sum(bnn[:i]) + edges[i][0][0] # source of e1
        v1 = sum(bnn[:i]) + edges[i][0][1] # dst of e1
        u2 = sum(bnn[:i]) + edges[i][1][0] # source of e2
        v2 = sum(bnn[:i]) + edges[i][1][1] # dst of e2
        
        tmscore=tmscores[i]
        
        z_e1 = (g.ndata['h'][u1]+g.ndata['h'][v1])/2 # edge 1 embed (mean)
        z_e2 = (g.ndata['h'][u2]+g.ndata['h'][v2])/2 # edge 2 embed (mean)
        
        #print('Two edges embeddings are ', z_e1,z_e2)
        
        loss += (F.pairwise_distance(z_e1,z_e2)-5*tmscore)**2 # todo
        
    return loss