# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:37:57 2020

@author: jacqu

Extract motifs graphs from RNA graphs and compute embeddings

"""

import pickle 
import networkx as nx 
import os 


graphs_dir = '../../data/motifs_graphs'
write_dir = '../../data/motifs_chunks'

with open('3dmotifs_dict.pickle', 'rb') as f :
    d = pickle.load(f)
    
for pdb, v in d.items():
    
    try:
        g0 = pickle.load(open(os.path.join(graphs_dir,pdb+'.pickle'), 'rb'))
        
    except:
        #print(pdb , ' not found')
        pass
    
    # nodes
    # Copy graph for each motif found in it 
    for i in range(len(v)):
        g=nx.MultiGraph(g0)
        bads = []
        for n, data in g.nodes(data=True):
            pos = data['nucleotide'].pdb_pos
            
            if(n[0]!=v[i][1] or pos not in v[i][2]):
                bads.append(n)
                
        g.remove_nodes_from(bads)
        
        print(g.number_of_nodes())
        
        if(g.number_of_nodes()>0):
            
            with open(os.path.join(write_dir,pdb+'_'+str(i)+'.pickle'), 'wb') as f:
                pickle.dump(g,f)
                pickle.dump(v[i][0],f)