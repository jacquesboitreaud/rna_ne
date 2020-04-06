# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset & Loader classes for RNA graphs with 3D-derived node features 
Used to compute pretrained node embeddings 

"""

import os 
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '../..'))
    
import torch
import dgl
import pandas as pd
import pickle
import numpy as np
import itertools
from collections import Counter
import networkx as nx

from tqdm import tqdm

# Do Not remove, required for loading pickle rna graphs
from rna_classes import *
from data_processing.graph_utils import *

from torch.utils.data import Dataset, DataLoader, Subset


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs (graph, pdb_id)
    #  (graph, context graph, node_idx, pair_label).
    graphs, pdbids = map(list, zip(*samples))
    
    try:
        batched_graph = dgl.batch(graphs)
    except: 
        print(graphs)
    
    return batched_graph, pdbids


class mgDataset(Dataset):
    """ 
    pytorch Dataset for training node classification for RNA binding 
    """
    def __init__(self, rna_graphs_path,
                 N_graphs,
                 emb_size,
                 attributes,
                 add_true_edges):
        
        self.path = rna_graphs_path
        
        if(N_graphs!=None):
            self.all_graphs = os.listdir(self.path)[:N_graphs] # Cutoff number
        else:
            self.all_graphs = os.listdir(self.path)
            np.random.seed(10)
            np.random.shuffle(self.all_graphs)
            
        self.n=len(self.all_graphs)
        
        # Params for getitem (training samples):
        self.emb_size = emb_size
        self.attributes = attributes
        
        # Wether to keep true edge labels in graph 
        # Build edge map
        self.true_edges=add_true_edges
        if(self.true_edges):
            print('Parsing true FR3D edge types in input graphs...')
            self.true_edge_map, self.true_edge_freqs = self._get_edge_data()
            self.num_true_edge_types = len(self.true_edge_map)
            print(f"found {self.num_true_edge_types} FR3D edge types, frequencies: {self.true_edge_freqs}")
        
        # the simplified labels to feed the GNN with (0,1)
        self.num_edge_types=2
        # Edge map with Backbone (0) and pairs (1)
        self.edge_map={'B35':0,
                  'B53':0}
        
        
    def _get_simple_etype(self,label):
        # Returns index of edge type for an edge label
        if(label in ['B35','B53']):
            return torch.tensor(0)
        else:
            return torch.tensor(1) # Non canonical edges category
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        
        # pick a graph (n°idx in the list)
        with open(os.path.join(self.path, self.all_graphs[idx]),'rb') as f:
            G = pickle.load(f)
            pdb = self.all_graphs[idx][:-7]
        
        G = nx.to_undirected(G)
        
        # Add one-hot edge types to features 
        one_hot = {edge: self._get_simple_etype(label) for edge, label in
       (nx.get_edge_attributes(G, 'label')).items()}
        nx.set_edge_attributes(G, name='one_hot', values=one_hot)
                
        if(self.true_edges): # add true edge types 
            true_ET = {edge: torch.tensor(self.true_edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(G, 'label')).items()}
            nx.set_edge_attributes(G, name='true_ET', values=true_ET)
        
        
        # Create dgl graph
        g_dgl = dgl.DGLGraph()

        # Add true edge types to features (for visualisation & clustering)
        if(self.true_edges):
            g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot','true_ET'], 
                                node_attrs = self.attributes+['Mg_binding'])
        else:
            g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot'], 
                                node_attrs = self.attributes+['Mg_binding'])
        
        # Init node embeddings with nodes features
        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata[a].view(-1,1) for a in self.attributes], dim = 1)
        
        # Return pair graph, pdb_id
        return g_dgl, pdb
    
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
                 attributes,
                 batch_size=32,
                 num_workers=0,
                 true_edges = True,
                 EVAL=False):
        
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 
        
        EVAL returns just the test loader 
        else, returns train, valid, 0

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = mgDataset(rna_graphs_path=path,
                                  N_graphs= N_graphs,
                                  emb_size=emb_size,
                                  attributes = attributes,
                                  add_true_edges = true_edges)
        self.num_edge_types = self.dataset.num_edge_types
        self.EVAL=EVAL
        
        print(f'***** {len(attributes)} node attributes will be used: {attributes}'  )

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 0.9, 1
        train_index, valid_index = int(split_train * n), int(split_valid * n)


        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        if(self.EVAL):
            train_set = Subset(self.dataset, train_indices[:1000]) # select just a small subset
        else:
            train_set = Subset(self.dataset, train_indices)
            
        valid_set = Subset(self.dataset, valid_indices)
        #test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")

        if(not self.EVAL): # Pretraining phase 
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
    
            valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                       num_workers=self.num_workers, collate_fn=collate_block)
            
            return train_loader, valid_loader, 0
        
        else: # Eval or visualization phase 
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            
            test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)

            return train_loader,0, test_loader
        
if __name__=='__main__':
    pass
            
            