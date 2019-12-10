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
import itertools
from collections import Counter

from rna_classes import *
import networkx as nx


from tqdm import tqdm
from rna_classes import *

from torch.utils.data import Dataset, DataLoader, Subset


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, edges, targets = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    
    bnn = batched_graph.batch_num_nodes
    N = len(bnn) # batch size
    edge_idces = torch.zeros((N,4), dtype=torch.long)
    for i in range(N):
        edge_idces[:,0] = torch.tensor([sum(bnn[:i]) + edges[i][0][0] for i in range(N)]) 
        # source of e1
        edge_idces[:,1] = torch.tensor([sum(bnn[:i]) + edges[i][0][1] for i in range(N)]) 
        # dst of e1
        edge_idces[:,2] = torch.tensor([sum(bnn[:i]) + edges[i][1][0] for i in range(N)]) 
        # source of e2
        edge_idces[:,3] = torch.tensor([sum(bnn[:i]) + edges[i][1][1] for i in range(N)]) 
        # dst of e2
    
    return batched_graph, edge_idces, targets


class rnaDataset(Dataset):
    """ 
    pytorch Dataset for training on pairs of nodes of RNA graphs 
    """
    def __init__(self, rna_graphs_path,
                 N_graphs,
                 emb_size,
                 simplified_edges,
                 debug=False):
        
        self.path = rna_graphs_path
        if(N_graphs!=None):
            self.all_graphs = os.listdir(self.path)[:N_graphs] # Cutoff number
        else:
            self.all_graphs = os.listdir(self.path)
            np.random.seed(10)
            np.random.shuffle(self.all_graphs)
            
        self.n = len(self.all_graphs)
        self.emb_size = emb_size
        # Build edge map
        self.simplified_edges=simplified_edges
        if(not self.simplified_edges):
            self.edge_map, self.edge_freqs = self._get_edge_data()
            self.num_edge_types = len(self.edge_map)
            print(f"found {self.num_edge_types} edge types, frequencies: {self.edge_freqs}")
        else:
            self.num_edge_types=4
            # Edge map with Backbone (0), WW (1), stackings (2) and others (3)
            self.edge_map={'B35':0,
                      'B53':0,
                      'CWW':1,
                      'S33':2,
                      'S35':2,
                      'S53':2,
                      'S55':2}
            print("Using simplified edge representations. 4 categories of edges")
        
        if(debug):
            # special case for debugging
            pass
        
    def _get_simple_etype(self,label):
        # Returns index of edge type for an edge label
        if(label in self.edge_map):
            return torch.tensor(self.edge_map[label])
        else:
            return torch.tensor(3) # Non canonical edges category
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # gets the RNA graph n°idx in the list
        with open(os.path.join(self.path, self.all_graphs[idx]),'rb') as f:
            graph = pickle.load(f)
            e1,e2, tmscore = pickle.load(f)
            
        e1_vertices=(e1[0][1], e1[1][1])
        e2_vertices=(e2[0][1], e2[1][1])
        e1 = [i for (i,n) in enumerate(graph.nodes()) if n[1] in e1_vertices]
        e2 = [i for (i,n) in enumerate(graph.nodes()) if n[1] in e2_vertices]
        
        graph = nx.to_undirected(graph)
        if(self.simplified_edges):
            one_hot = {edge: self._get_simple_etype(label) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}
        else:
            one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}

        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
        
        # Create dgl graph
        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
        
        g_dgl.ndata['h'] = torch.ones((g_dgl.number_of_nodes(), self.emb_size)) # nodes embeddings 
        
        return g_dgl,(e1,e2), tmscore
    
    def _get_edge_data(self):
        """
        Get edge type statistics, and edge map.
        """
        edge_counts = Counter()
        print("Collecting edge data...")
        graphlist = os.listdir(self.path)
        for g in tqdm(graphlist):
            graph = pickle.load(open(os.path.join(self.path, g), 'rb'))
            edges = {e_dict['label'] for _,_,e_dict in graph.edges(data=True)}
            edge_counts.update(edges)
            
        # Edge map with all different types of edges (FR3D edges)
        edge_map = {label:i for i,label in enumerate(sorted(edge_counts))}
        
        IDF = {k: np.log(len(graphlist)/ edge_counts[k] + 1) for k in edge_counts}
        return edge_map, IDF
        
    
class Loader():
    def __init__(self,
                 path,
                 N_graphs,
                 emb_size,
                 batch_size=64,
                 num_workers=4,
                 debug=False,
                 simplified=True,
                 EVAL=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 
        
        EVAL returns just the test loader 
        else, returns train, valid, 0

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = rnaDataset(rna_graphs_path=path,
                                  N_graphs= N_graphs,
                                  emb_size=emb_size,
                                  debug=debug,
                                  simplified_edges=simplified)
        self.num_edge_types = self.dataset.num_edge_types
        self.EVAL=EVAL

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 0.8, 0.9
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        
        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")

        if(not self.EVAL):
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
    
            valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                       num_workers=self.num_workers, collate_fn=collate_block)
            
            return train_loader, valid_loader, 0
        
        else:
            test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)


            return 0,0, test_loader