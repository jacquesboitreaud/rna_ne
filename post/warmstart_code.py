# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:05:57 2020

@author: jacqu
"""


# Get edge map and freqs (same as I used )

with open('fr3d_edge_map.pickle','rb') as f:
    self.true_edge_map = pickle.load(f)
    self.edge_freqs = pickle.load(f)

# Getitem in dataloader : 

def __getitem__(self, idx):
    
    # pick a graph (n°idx in the list)
    with open(os.path.join(self.path, self.all_graphs[idx]),'rb') as f:
        G = pickle.load(f)
        pdb = self.all_graphs[idx][:-7]
    
    G = nx.to_undirected(G)
    
    # Add one-hot edge types to features 
    true_ET = {edge: torch.tensor(self.true_edge_map[label]) for edge, label in
           (nx.get_edge_attributes(G, 'label')).items()}
    nx.set_edge_attributes(G, name='one_hot', values=true_ET)
        
    # Create dgl graph
    g_dgl = dgl.DGLGraph()

    # Add true edge types to features (for visualisation & clustering)
    g_dgl.from_networkx(nx_graph=G, edge_attrs=['one_hot'], 
                            node_attrs = self.attributes)
    
    # Init node embeddings with nodes features
    floatid = g_dgl.ndata['identity'].float()
    g_dgl.ndata['h'] = torch.cat([g_dgl.ndata['angles'], floatid], dim = 1)
    
    # Return pair graph, pdb_id
    return g_dgl, pdb