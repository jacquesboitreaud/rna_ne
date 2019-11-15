# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class for pairs of nodes and RNA graphs handling

TODO: Collate block, loader + change paths to files 
"""

import os 
import sys
if __name__ == "__main__":
    sys.path.append("..")
    
import torch
import dgl
import pandas as pd
import pickle
import numpy as np
import random
import itertools
from collections import Counter

from rna_classes import *
import networkx as nx


from tqdm import tqdm
from rna_classes import *
from graph_process import *

from torch.utils.data import Dataset, DataLoader, Subset


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, n_nodes, node_indices, targets = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    
    cpt=0
    idces=[]
    for i in range(len(samples)):
        n1, n2 = node_indices[i][0]-1, node_indices[i][1]-1 # Nodes are indexed from 1 (to check...)
        idces.append((n1+cpt,n2+cpt)) # find node indices in the batched_graph object
        cpt+=n_nodes[i]
    
    return batched_graph, idces, targets


class rnaDataset(Dataset):
    """ 
    pytorch Dataset for training on pairs of nodes of RNA graphs 
    """
    def __init__(self, emb_size=1,
                rna_graphs_path="../data/annotated",
                debug=False, shuffled=False):
        
        self.path = rna_graphs_path
        self.all_graphs = os.listdir(self.path)
        self.n = len(self.all_graphs)
        self.emb_size = emb_size
        # Build edge map
        self.edge_map, self.edge_freqs = self._get_edge_data()
        self.num_edge_types = len(self.edge_map)
        print(f"found {self.num_edge_types} edge types, frequencies: {self.edge_freqs}")
        
        if(debug):
            # special case for debugging
            pass
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # gets the RNA graph n°idx in the list
        # Annotated pickle files are tuples (g, dict of rmsds between nodepairs)
        
        graph, pairwise_dists = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))
        
        graph = nx.to_undirected(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}

        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
        
        # Create dgl graph
        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])

        n_nodes = len(g_dgl.nodes())
        g_dgl.ndata['h'] = torch.ones((n_nodes, self.emb_size)) # nodes embeddings 
        
        # Random selection of a pair of nodes and their rmsd 
        nodes, r = random.choice(list(pairwise_dists.items()))
        
        #TODO : find a way to pass this info batchwise
        # K = triplet n_nodes, n1,n2
        n1,n2= nodes[0][1],nodes[1][1]
        
        return g_dgl, n_nodes, (n1,n2), r
    
    def _get_edge_data(self):
        """
        Get edge type statistics, and edge map.
        """
        edge_counts = Counter()
        edge_labels = set()
        print("Collecting edge data...")
        graphlist = os.listdir(self.path)
        for g in tqdm(graphlist):
            graph, _ = pickle.load(open(os.path.join(self.path, g), 'rb'))
            edges = {e_dict['label'] for _,_,e_dict in graph.edges(data=True)}
            edge_counts.update(edges)
        
        # Edge map with all different types of edges (FR3D edges)
        edge_map = {label:i for i,label in enumerate(sorted(edge_counts))}
        
        # Edge map with Backbone (0), WW (1), and others (2)
        #TODO
        IDF = {k: np.log(len(graphlist)/ edge_counts[k] + 1) for k in edge_counts}
        return edge_map, IDF
        
    
class Loader():
    def __init__(self,
                 annot_path='../data/annotated',
                 emb_size=1,
                 batch_size=128,
                 num_workers=20,
                 debug=False,
                 shuffled=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = rnaDataset(emb_size, rna_graphs_path=annot_path,
                          debug=debug,
                          shuffled=shuffled)
        self.num_edge_types = self.dataset.num_edge_types

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 0.7, 0.7
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        
        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")


        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_block)

        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)


        # return train_loader, valid_loader, test_loader
        return train_loader, 0, test_loader