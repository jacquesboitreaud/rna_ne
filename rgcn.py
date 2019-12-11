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


    def forward(self, g, edge_idces):
        #print('edge data size ', g.edata['one_hot'].size())
        
        u1=edge_idces[:,0]
        v1=edge_idces[:,1]
        u2=edge_idces[:,2]
        v2=edge_idces[:,3]
        
        for layer in self.layers:
             #print(g.ndata['h'].size())
             #print(g.edata['one_hot'].size())
             g.ndata['h']=layer(g,g.ndata['h'],g.edata['one_hot'])
        
        g.ndata['h']=self.dense(g.ndata['h'])
        return (g.ndata['h'][u1]+g.ndata['h'][v1])/2, (g.ndata['h'][u2]+g.ndata['h'][v2])/2
    
    def draw_rec(self, z_e1,z_e2, tmscores, title = ''):
        """
        A way to assess how the loss fits the TM scores task visually
        :param true_K:
        :param predicted_K:
        :param loss_value: python float
        :return:
        """
        true_K = 5*(1-tmscores)
        predicted_K = torch.sqrt(torch.sum((z_e1-z_e2)**2,dim=1)).view(-1,1)
        true_K=true_K.cpu()
        predicted_K= predicted_K.cpu()
        
        #print(true_K)
        #print(predicted_K)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(true_K.detach().numpy(), vmin=0, vmax=1, ax=ax1, square=True, cbar=False)
        sns.heatmap(predicted_K.detach().numpy(), vmin=0, vmax=1, ax=ax2, square=True, cbar=False,
                    cbar_kws={"shrink": 1})
        ax1.set_title("Ground Truth")
        ax2.set_title("GCN")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        
def Loss(z_e1,z_e2, tmscores, v=False):
    # Takes batches graph and labels, computes loss 
    #print('Two edges embeddings are ', z_e1,z_e2)
    predicted_K= torch.sqrt(torch.sum((z_e1-z_e2)**2,dim=1)).view(-1,1)
    
    #true_K = 5*(1-tmscores) # linear scaling ! 
    true_K = tmscores.pow_(-1) # 1/ tm squared !  
    if(v):
        print('Predicted K: ', predicted_K)
        print('True K: ', true_K)
    loss = torch.mean(torch.sqrt((predicted_K-true_K)**2)) 
        
    return loss

