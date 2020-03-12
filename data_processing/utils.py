# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:29:14 2019

@author: jacqu

Utils for pdb rna graphs
"""

import itertools
import networkx as nx
import numpy as np
import dgl
import torch

#faces = ['W', 'S', 'H']
#orientations = ['C', 'T']
#valid_edges = set(['B53'] + [orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations])

def remove_self_edges(G):
    to_drop=[]
    for e in G.edges():
        if(e[0]==e[1]):
            to_drop.append(e)
    G.remove_edges_from(to_drop)
    
def label_edges(G):
    #TODO 3 categories of labels: Backbone, canonical & other 
    labels = np.zeros(4)
    return labels

def knbrs(G, start, k):
    # Get nodes reachable at a distance of k from 'start' node
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g




