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

import seaborn as sns 
import matplotlib.pyplot as plt

class RGCN(nn.Module):
    # Computes embeddings for all nodes
    # No features
    def __init__(self, features_dim, h_dim, out_dim , num_rels, num_bases=-1, num_layers=1, dropout = 0.2):
        super(RGCN, self).__init__()
        
        self.features_dim, self.h_dim, self.out_dim = features_dim, h_dim, out_dim
        self.num_layers = num_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        
        self.d = dropout
        
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        if(self.num_layers==1):
            i2h = RelGraphConv(self.features_dim, self.out_dim, self.num_rels, num_bases = self.num_bases, dropout = self.d)
        else:
            i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, num_bases = self.num_bases,
                               activation=nn.ReLU(), dropout = self.d)
        self.layers.append(i2h)
        
        # hidden to hidden
        if(self.num_layers>2):
            for _ in range(self.num_layers-2):
                h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, num_bases = self.num_bases,
                                   activation=nn.ReLU(), dropout = self.d)
                self.layers.append(h2h)
                
        # hidden to output
        if(self.num_layers>=2):
            h2o = RelGraphConv(self.h_dim, self.out_dim, self.num_rels, num_bases = self.num_bases, dropout = self.d) 
            self.layers.append(h2o)

    def forward(self, g):
        
        for layer in self.layers:
             g.ndata['h']=layer(g,g.ndata['h'],g.edata['one_hot'])
            
        return g.ndata['h']
        
class Model(nn.Module):
    """ Model instance that contains a GNN and a context GNN """
    
    def __init__(self, features_dim, h_dim, out_dim , num_rels, radii_params, num_bases=-1, dropout = 0.2):
        super(Model, self).__init__()
        
        self.features_dim, self.h_dim, self.out_dim = features_dim, h_dim, out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        
        self.K, self.r1, self.r2 = radii_params
        self.d = dropout 
        
        self.cgnn_layers = int(self.r2-self.r1)
        assert(self.cgnn_layers==self.K) # Patch representations of radius 1 , context ring of width 1 
        print(' ************* Model initialisation *******************')
        print(f'gnn has {self.K} layers ')
                
        # create rgcn layers
        self.build_model()

    def build_model(self):
        
        self.GNN = RGCN(self.features_dim,self.h_dim, self.out_dim,self.num_rels,
                        self.num_bases, num_layers=self.K, dropout = self.d)
        
        self.linear=nn.Linear(self.out_dim, self.out_dim, bias=False)
        
    def forward(self, g, ctx_g):
        # Forward pass of both GNNs; embeddings stored in g.ndata['h']
        self.GNN(g)
        self.GNN(ctx_g)
        
    def linear_tf(self,h):
        # Learnable linear transform of embeddings before applying contrastive loss
        return self.linear(h)

        
def draw_rec( prod, label, title = ''):
        """
        A way to assess how the loss fits the TM scores task visually
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(label.cpu().detach().numpy(), vmin=0, vmax=1, ax=ax1, square=True, cbar=False)
        sns.heatmap(prod.cpu().detach().numpy(), vmin=0, vmax=1, ax=ax2, square=True, cbar=False,
                    cbar_kws={"shrink": 1})
        ax1.set_title("Ground Truth")
        ax2.set_title("GCN")
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
        
def pretrainLoss(h_v, h_ctx, label, v=False, show = False):
    # Context prediction loss
    #prod = torch.sigmoid(torch.bmm(h_v.unsqueeze(1),h_ctx.unsqueeze(2)).squeeze())
    prod = torch.bmm(h_v.unsqueeze(1),h_ctx.unsqueeze(2)).squeeze().view(-1,1)
    label = label.view(-1,1)
    if(v):
        print('Dot-product: ', torch.sigmoid(prod))
        print('Truth : ', label)
    if(show):
        draw_rec(torch.sigmoid(prod), label)
        plt.show()
    loss = F.binary_cross_entropy_with_logits(prod, label, reduction = 'sum') 
     
    return loss, prod

