# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

GCN to learn node embeddings on RNA graphs 


"""


import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class Net(torch.nn.Module):
    
    def __init__(self, num_node_features, embedding_dim):
        super(Net, self).__init__()
        
        # GCN layers : node embeddings 
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, embedding_dim)
        
        # Aggregate all node embeddings : n_nodes -> 1 
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x= global_mean_pool(x, data.batch)

        return F.log_softmax(x, dim=1)
    
def regLoss(out, target):
    """ MSE regression loss """
    return F.mse_loss(out, target, reduction='sum')




def Loss(z1, z2, rmsd):
    """ 
    Loss function to force d(z1,z2) proportional to neighborhoods rmsd
    """
    BCE = F.binary_cross_entropy(out, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    error= F.mse_loss(pred_properties, y, reduction="sum")
    total_loss=BCE + kappa*KLD + error
        
    return total_loss, BCE, KLD, error # returns 4 values
