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
    graphs, ctx_graphs, u_idx, labels = map(list, zip(*samples))
    
    try:
        batched_graph = dgl.batch(graphs)
    except: 
        print(graphs)
    ctx_batched_graph = dgl.batch(ctx_graphs)
    labels = torch.tensor(labels, dtype = torch.float)
    
    return batched_graph, ctx_batched_graph, u_idx, labels


class pretrainDataset(Dataset):
    """ 
    pytorch Dataset for training on pairs of nodes of RNA graphs 
    """
    def __init__(self, rna_graphs_path,
                 N_graphs,
                 emb_size,
                 radii_params,
                 attributes,
                 simplified_edges,
                 EVAL,
                 debug,
                 fix_seed):
        
        self.EVAL=EVAL
        self.debug = debug
        if(self.debug):
            print('Debug prints set to True')
        self.fix_seed = fix_seed
        self.path = rna_graphs_path
        
        if(N_graphs!=None):
            self.all_graphs = os.listdir(self.path)[:N_graphs] # Cutoff number
        else:
            self.all_graphs = os.listdir(self.path)
            np.random.seed(10)
            np.random.shuffle(self.all_graphs)
            
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
            self.num_edge_types=2
            print(f"Using simplified edge representations. {self.num_edge_types} categories of edges")
            # Edge map with Backbone (0) and pairs (1)
            #TODO handle stackings S33 S35 S53 S55
            self.edge_map={'B35':0,
                      'B53':0}
        
    def _get_simple_etype(self,label):
        # Returns index of edge type for an edge label
        if(label in ['B35','B53']):
            return torch.tensor(0)
        else:
            return torch.tensor(1) # Non canonical edges category
            
    def __len__(self): # Number of samples in epoch : should be >> n_graphs (1 sample = 1 node)
        return self.n_graphs *50
    
    def __getitem__(self, idx):
        
        #Fix random seed for all other samplings 
        if(self.fix_seed):
            np.random.seed(10)
        
        # pick a graph at random 
        gidx = np.random.randint(self.n_graphs)
        gid = self.all_graphs[gidx]
        
        with open(os.path.join(self.path, gid),'rb') as f:
            G = pickle.load(f)
        G.to_undirected() # Undirected graph 
         
        # Pick a node at random : 
        N = G.number_of_nodes()
        u_idx = np.random.randint(N)
        
        # Selected node has idx u_idx in sorted(G.nodes) // coherent with dgl reindexing
        u = sorted(G.nodes)[u_idx]
        
        # Random sample positive or negative context 'deterministic but 'idx' unused elsewhere
        r = int(idx%2==0)
        
        if(r>0.5): # positive context 
            G_ctx = nx.Graph(G)
            assert(not G_ctx.is_directed())

            anchor_nodes = [n for n in G_ctx.neighbors(u)]
            pair_label = 1 # positive pair 
            ctx_nodes = nodes_within_radius(G_ctx, u_idx, inner=self.r1, outer=self.r2)
            if(len(ctx_nodes)==0):
                assert(False), f'graph id {gid}, node {u}, positive pair, context has size zero'
                
            G_ctx.remove_nodes_from([n for n in G_ctx if n not in set(ctx_nodes)])
            
            
        else:
            neg = np.random.randint(self.n_graphs)
            ngid = self.all_graphs[neg]
            with open(os.path.join(self.path, ngid),'rb') as f:
                G_ctx = pickle.load(f)
            G_ctx = nx.to_undirected(G_ctx)
            N = G_ctx.number_of_nodes()
            
            assert(not G_ctx.is_directed())
            
            u_neg_idx = np.random.randint(N)
            
            G_ctx = nx.Graph(G_ctx)
            u_neg = sorted(G_ctx.nodes)[u_neg_idx]
            anchor_nodes = [n for n in G_ctx.neighbors(u_neg)]
            ctx_nodes = nodes_within_radius(G_ctx, u_neg_idx, inner=self.r1, outer=self.r2)
            if(len(ctx_nodes)==0):
                assert(False), f'graph id {ngid}, node {u_neg}, negative pair, context has size zero'
            G_ctx.remove_nodes_from([n for n in G_ctx if n not in set(ctx_nodes)])
            
            pair_label = 0 # negative pair 
            
        # Add anchor nodes as a node feature 
        is_anchor = {node: torch.tensor(node in anchor_nodes) for node in G_ctx.nodes()}
        nx.set_node_attributes(G_ctx, name='anchor', values = is_anchor)
            
        
        # Cut graph to radius K around node u 
        G=nx.Graph(G)
        assert(not G.is_directed())
        local_nodes = nodes_within_radius(G, u_idx, inner=0, outer=self.K)
        
        G.remove_nodes_from([n for n in G if n not in set(local_nodes)])
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

        try: # Catching a weird bug 
            g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot'], node_attrs = self.attributes)
            ctx_g_dgl.from_networkx(nx_graph=G_ctx, edge_attrs=['one_hot'], node_attrs = self.ctx_attributes)
        except:
            for a in self.attributes:
                print('***** debug : context graph node attrs *****')
                print(nx.get_node_attributes(G_ctx,a))
                print('***** debug : patch graph node attrs *****')
                print(nx.get_node_attributes(G,a))
            
        # Init node embeddings with nodes features
        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata[a].view(-1,1) for a in self.attributes], dim = 1)
        ctx_g_dgl.ndata['h'] = torch.cat([ctx_g_dgl.ndata[a].view(-1,1) for a in self.attributes], dim=1)
        
        
        if(self.EVAL): #TODO
            raise NotImplementedError
            labels = 0
            return g_dgl, labels
        else:
            return g_dgl, ctx_g_dgl, u_idx, pair_label
    
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
                 radii_params, 
                 attributes,
                 batch_size=32,
                 num_workers=0,
                 debug=False,
                 simplified_edges=True,
                 EVAL=False, 
                 fix_seed=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 
        
        EVAL returns just the test loader 
        else, returns train, valid, 0

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = pretrainDataset(rna_graphs_path=path,
                                  N_graphs= N_graphs,
                                  emb_size=emb_size,
                                  radii_params = radii_params,
                                  attributes = attributes,
                                  EVAL=EVAL,
                                  debug=debug,
                                  simplified_edges=simplified_edges, 
                                  fix_seed = fix_seed)
        self.num_edge_types = self.dataset.num_edge_types
        self.EVAL=EVAL
        
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
        
        if(self.EVAL):
            train_set = Subset(self.dataset, train_indices[:1000]) # select just a small subset
        else:
            train_set = Subset(self.dataset, train_indices)
            
        valid_set = Subset(self.dataset, valid_indices)
        #test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")

        if(not self.EVAL): # Pretraining phase : only train loader 
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            
            return train_loader, 0, 0
        
        else: # Eval or visualization phase 
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            
            test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)


            return train_loader,0, test_loader
        
if __name__=='__main__':
    
    l = Loader(path ='../data/chunks', 
               N_graphs = 100,
               emb_size = 12, 
               radii_params = (1,1,3),
               attributes = ['delta','chi', 'gly_base'])
    
    
    g = l.dataset.__getitem__(1)
    
    graph = g[0].to_networkx()
            
            