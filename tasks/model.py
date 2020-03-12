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
    def __init__(self, features_dim, h_dim, out_dim , num_rels, num_bases=-1, num_layers=1, pool=False):
        super(RGCN, self).__init__()
        
        self.features_dim, self.h_dim, self.out_dim = features_dim, h_dim, out_dim
        self.num_layers = num_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.pool=pool
        
        if(self.pool):
            self.pooling_layer = SumPooling()
        
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        if(self.num_layers==1):
            i2h = RelGraphConv(self.features_dim, self.out_dim, self.num_rels) #, activation=nn.ReLU())
        else:
            i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
        self.layers.append(i2h)
        
        # hidden to hidden
        if(self.num_layers>2):
            for _ in range(self.num_layers-2):
                h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
                self.layers.append(h2h)
        # hidden to output
        if(self.num_layers>=2):
            h2o = RelGraphConv(self.h_dim, self.out_dim, self.num_rels) #, activation=nn.ReLU())
            self.layers.append(h2o)


    def forward(self, g):
        #print('edge data size ', g.edata['one_hot'].size())
        #print('node data size ', g.ndata['h'].size())
        #print('initial h: ', g.ndata['h'])
        
        for layer in self.layers:
             g.ndata['h']=layer(g,g.ndata['h'],g.edata['one_hot'])
            
        if(self.pool):
            out = self.pooling_layer(g, g.ndata['h'])
            return out 
        else:
            return g.ndata['h']

        
def draw_rec( prod, label, title = ''):
        """
        A way to assess how the loss fits the TM scores task visually
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(label.detach().numpy(), vmin=0, vmax=1, ax=ax1, square=True, cbar=False)
        sns.heatmap(prod.detach().numpy(), vmin=0, vmax=1, ax=ax2, square=True, cbar=False,
                    cbar_kws={"shrink": 1})
        ax1.set_title("Ground Truth")
        ax2.set_title("GCN")
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
        
def classifLoss(h, labels, show = False):
    # Node classification loss 

    labels = labels.view(-1,1)

    if(show):
        draw_rec(torch.sigmoid(h), labels)
        plt.show()
    loss = F.binary_cross_entropy_with_logits(h, labels, reduction = 'sum') 
     
    return loss

