# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Parse graphs looking for nucleotides involved in a pairing . 

"""


import numpy as np
import pickle 
import os 
import networkx as nx
import sys
import argparse


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

    from data_processing.graph_utils import *
    from data_processing.angles import base_angles
    from data_processing.rna_classes import *
    from data_processing.utils import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing already processed graphs", 
                        type=str, default="../data/chunks")
    
     # =======

    args=parser.parse_args()
    
    # Hyperparams 
    gr_dir = args.graphs_dir
    
    large_small = pickle.load(open('large_graphs.pickle','rb'))
    print(len(large_small), ' ignored graphs / too big or too small')
    
    d={}
    cpt=0
    nodes_cpt, good_nodes_cpt = 0,0
    
    for pdb_id in os.listdir(gr_dir):
        
        if(pdb_id in large_small):
            print('passing graph')
            continue
        
        cpt+=1
        
        if(cpt%100)==0:
            print(cpt)
         
        # Load graph  
        g = pickle.load(open(os.path.join(gr_dir,pdb_id), 'rb'))
        d[pdb_id]=[]
        
        nodes =g.nodes(data=True)
        N = g.number_of_nodes()
        nodes_cpt +=N
         
        for u,v,data in g.edges(data=True):
            label = data['label']
            
            if(label not in {'B35', 'B53','S33', 'S35','S53','S55'}):
                
                if(u not in d[pdb_id]):
                    d[pdb_id].append(u)
                if(v not in d[pdb_id]):
                    d[pdb_id].append(v)
                
        good_nodes_cpt +=len(d[pdb_id])
        
        
print(cpt, ' graphs parsed')             
print(good_nodes_cpt, ' good nodes')
print(nodes_cpt, ' overall nodes read')

with open('selected_nodes.pickle','wb') as f:
    pickle.dump(d,f)
