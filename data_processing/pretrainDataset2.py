# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset + loader class for RNA graphs pretraining using context prediction (https://arxiv.org/abs/1905.12265)

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
import random

from tqdm import tqdm

# Do Not remove, required for loading pickle rna graphs
import rna_classes
from rna_classes import *
from graph_utils import *

from torch.utils.data import Dataset, DataLoader, Subset


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs
    #  (graph, context graph, node_idx, pair_label).
    graphs, ctx_graphs, u_idx= map(list, zip(*samples))
    
    try:
        batched_graph = dgl.batch(graphs)
    except: 
        print(graphs)
    ctx_batched_graph = dgl.batch(ctx_graphs)
    
    return batched_graph, ctx_batched_graph, u_idx


class pretrainDataset(Dataset):
    """ 
    pytorch Dataset for training on pairs of nodes of RNA graphs 
    """
    def __init__(self, graphs_path,
                 nodes_dict,
                 N_graphs,
                 emb_size,
                 radii_params,
                 attributes,
                 simplified_edges,
                 debug,
                 fix_seed):
        
        self.debug = debug
        if(self.debug):
            print('Debug prints set to True')
        self.fix_seed = fix_seed
        
        self.path = graphs_path
        self.nodes_dict = nodes_dict 
        
        if(N_graphs!=None):
            self.all_graphs = sorted(self.nodes_dict)[:N_graphs] # Cutoff number
        else:
            self.all_graphs = sorted(self.nodes_dict)
            
        self.n_graphs=len(self.all_graphs)
        
        # Params for getitem (training samples):
        self.emb_size = emb_size
        self.K, self.r1, self.r2 = radii_params
        self.attributes = attributes
        self.ctx_attributes = self.attributes + ['anchor']
        
        
        # Build edge map
        self.simplified_edges=simplified_edges
        if(not self.simplified_edges):
            self.edge_map, self.edge_freqs = self._get_edge_data()
            self.num_edge_types = len(self.edge_map)
            print(f"found {self.num_edge_types} edge types, frequencies: {self.edge_freqs}")
        else:
            self.num_edge_types=3
            print(f"Using simplified edge representations. {self.num_edge_types} categories of edges")
            # Edge map with Backbone (0) and pairs (1)
            #TODO handle stackings S33 S35 S53 S55
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
            
    def __len__(self): # Number of samples in epoch : should be >> n_graphs (1 sample = 1 node)
        return 40
    
    def __getitem__(self, idx):
        
        #Fix random seed for all other samplings 
        if(self.fix_seed):
            random.seed(10)
        
        # pick a graph  and node at random 
        gid = random.choice(self.all_graphs)
        u = random.choice(self.nodes_dict[gid])
        
        with open(os.path.join(self.path, gid),'rb') as f:
            G = pickle.load(f)
        G.to_undirected() # Undirected graph
        
        # ============= Context graph ===================================
        G_ctx = nx.MultiGraph(G)
        assert(not G_ctx.is_directed())

        anchor_nodes = [n for n in G_ctx.neighbors(u)]
        
        ctx_nodes = nodes_within_radius(G_ctx, u, inner=self.r1, outer=self.r2)
        if(len(ctx_nodes)==0):
            assert(False), f'graph id {gid}, node {u}, positive pair, context has size zero'
            
        G_ctx.remove_nodes_from([n for n in G_ctx if n not in set(ctx_nodes)])
            
        # Add anchor nodes as a node feature 
        is_anchor = {node: float(node in anchor_nodes) for node in G_ctx.nodes()}
        nx.set_node_attributes(G_ctx, name='anchor', values = is_anchor)
            
        # ====================== Patch graph ===============================
        
        # Cut graph to radius K around node u : K=1, neighbours of u 
        G=nx.MultiGraph(G)
        assert(not G.is_directed())
        patch_nodes = [n for n in G.neighbors(u)] 
        patch_nodes.append(u)
        
        G.remove_nodes_from([n for n in G if n not in set(patch_nodes)])
        if(G.number_of_nodes()==0):
            assert(False), f'graph id {gid}, node {u}, no nodes left within radius K'
        # Get the index of node u in the new cropped graph
        nodes = sorted(G.nodes)
        u_idx = [i for i,n in enumerate(nodes) if n==u][0]
        
        # Add Edge types to features 
        if(self.simplified_edges):
            one_hot = {edge: self._get_simple_etype(label) for edge, label in
                   (nx.get_edge_attributes(G, 'label')).items()}
            one_hot_ctx = {edge: self._get_simple_etype(label) for edge, label in
                   (nx.get_edge_attributes(G_ctx, 'label')).items()}
        else:
            one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(G, 'label')).items()}
            one_hot_ctx = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(G_ctx, 'label')).items()}
        
        
        nx.set_edge_attributes(G, name='one_hot', values=one_hot)
        nx.set_edge_attributes(G_ctx, name='one_hot', values=one_hot_ctx)
        
        # Create dgl graph
        g_dgl = dgl.DGLGraph()
        ctx_g_dgl = dgl.DGLGraph()
        
        # Dgl graph build

        g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot'], node_attrs = self.attributes)
        try:
            ctx_g_dgl.from_networkx(nx_graph=G_ctx, edge_attrs=['one_hot'], node_attrs = self.ctx_attributes)
        except:
            print('ctx graph to dgl error')
            print('graph : ', gid, 'node', u)
            for a in self.ctx_attributes:
                print(nx.get_node_attribute(G_ctx,a))
            
        # Init node embeddings with nodes features
        #if('identity') in self.attributes:
        floatid = g_dgl.ndata['identity'].float()
        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata['angles'], floatid], dim = 1)
        floatid = ctx_g_dgl.ndata['identity'].float()
        ctx_g_dgl.ndata['h'] = torch.cat([ctx_g_dgl.ndata['angles'],floatid], dim=1)
        
        
        return g_dgl, ctx_g_dgl, u_idx
    
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
                 nodes_dict,
                 N_graphs,
                 emb_size,
                 radii_params, 
                 attributes,
                 batch_size=32,
                 num_workers=0,
                 debug=False,
                 simplified_edges=True,
                 fix_seed=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = pretrainDataset(graphs_path=path, 
                                       nodes_dict = nodes_dict ,
                                  N_graphs= N_graphs,
                                  emb_size=emb_size,
                                  radii_params = radii_params,
                                  attributes = attributes,
                                  debug=debug,
                                  simplified_edges=simplified_edges, 
                                  fix_seed = fix_seed)
        self.num_edge_types = self.dataset.num_edge_types
        
        print(f'***** {len(attributes)} node attributes will be used: {attributes}'  )

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))

        split_train, split_valid = 1, 1
        train_index, valid_index = int(split_train * n), int(split_valid * n)


        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        

        train_set = Subset(self.dataset, train_indices)
        
        #test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")

        # Pretraining phase : only train loader 
        train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block, pin_memory=True)
            
        return train_loader, 0, 0
        
if __name__=='__main__':
    pass
        
            
            