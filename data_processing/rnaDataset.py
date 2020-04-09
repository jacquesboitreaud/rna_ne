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
    sys.path.append(os.path.join(script_dir, '..'))
    
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
import rna_classes
from rna_classes import *
from graph_utils import *

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


class rnaDataset(Dataset):
    """ 
    pytorch Dataset for training on pairs of nodes of RNA graphs 
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
            print(len(self.all_graphs))
            
        self.n=len(self.all_graphs)
        
        # Params for getitem (training samples):
        self.emb_size = emb_size
        self.attributes = attributes
        
        # Wether to keep true edge labels in graph 
        # Build edge map
        self.true_edges=add_true_edges
        
        if(self.true_edges):
            print('Parsing true FR3D edge types in input graphs...')
            with open('fr3d_edge_map.pickle','rb') as f:
                
                try:
                    self.true_edge_map = pickle.load(f)
                    self.true_edge_freqs = pickle.load(f)
                except:
                    print('>>> Edge map and frequencies not found. Parsing the dataset...')
                    self.true_edge_map, self.true_edge_freqs = self._get_edge_data()
                
            self.num_edge_types = len(self.true_edge_map)
            print(f"found {self.num_edge_types} FR3D edge types, frequencies: {self.true_edge_freqs}")
            
        else:
            # the simplified edge labels to feed the GNN 
            self.num_edge_types=3
            # Edge map with Backbone (0) , stackings (1) and pairs (2)
            self.edge_map={'B35':0,
                          'B53':0,
                          'S33':1,
                          'S35':1,
                          'S53':1,
                          'S55':1}
        
        
    def _get_simple_etype(self,label):
        # Returns index of edge type for an edge label
        if(label in ['B35','B53']):
            return torch.tensor(0)
        elif(label in ['S33' ,'S35' ,'S53', 'S55']):
            return torch.tensor(1) 
        else:   # Base interaction edges category
            return torch.tensor(2)
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        
        # pick a graph (n°idx in the list)
        with open(os.path.join(self.path, self.all_graphs[idx]),'rb') as f:
            G = pickle.load(f)
            pdb = self.all_graphs[idx][:-7]
        
        G = nx.to_undirected(G)
        
        # Add simplified edge types to features 
        simple_one_hot = {edge: self._get_simple_etype(label) for edge, label in
       (nx.get_edge_attributes(G, 'label')).items()}
        nx.set_edge_attributes(G, name='one_hot', values=simple_one_hot)
                
        if(self.true_edges): # add true edge types 
            true_ET = {edge: torch.tensor(self.true_edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(G, 'label')).items()}
            nx.set_edge_attributes(G, name='true_ET', values=true_ET)
        
        # Create dgl graph
        g_dgl = dgl.DGLGraph()

        # Add true edge types to features (for visualisation & clustering)
        if(self.true_edges):
            g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot','true_ET'], node_attrs = self.attributes)
        else:
            g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot'], node_attrs = self.attributes)
        
        # Init node embeddings with nodes features
        floatid = g_dgl.ndata['identity'].float()
        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata['angles'], floatid], dim = 1)
        
        # Return pair graph, pdb_id + 'label' attribute to carry along
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
                 true_edges = True):
        
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
                                  attributes = attributes,
                                  add_true_edges = true_edges)
        self.num_edge_types = self.dataset.num_edge_types
        
        print(f'***** {len(attributes)} node attributes will be used: {attributes}'  )

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 1, 1
        train_index, valid_index = int(split_train * n), int(split_valid * n)


        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        train_set = Subset(self.dataset, train_indices)
            
        #test_set = Subset(self.dataset, test_indices)
        print(f"Loaded dataset contains {len(train_set)} samples")

        dataset_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            
        return dataset_loader, 0, 0
        
if __name__=='__main__':
    pass
            
            