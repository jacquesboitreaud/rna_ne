# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Reads RNA graphs annotated with FR3D edge labels 
Computes 3D-distance for edge pairs; 

Saves processed graph and dict of RMSDs to pickle file in 'annotated_dir'

DEBUG: on 6n2v pdb (contains only RNA and ligands)

        
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

    from data_processing.pdb_utils import *
    from data_processing.angles import base_angles
    from data_processing.rna_classes import *
    from data_processing.utils import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing 'rna_classes' nx graphs ", 
                        type=str, default="C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs")
    parser.add_argument('-c', "--cutoff", help="Max number of train samples. Set to -1 for all graphs in dir", 
                        type=int, default=100)
    parser.add_argument('-o', '--write_dir', help="path to directory to write preprocessed graphs ", 
                        type=str, default="../data/chunks")
    
     # =======

    args=parser.parse_args()
    
    # Hyperparams 
    gr_dir = args.graphs_dir
    annot_dir = args.write_dir
    
    angles = ['chi', 'delta', 'gly_base']
    
    print(f'Calculating {len(angles)} angles for each nt.')
    print(f'Graphs with node features will be saved to {annot_dir}')
    
    cpt=0
    
    for pdb_id in os.listdir(gr_dir):
        
        # Dict for new node attributes 
        d = {}
        for a in angles:
            d[a]={} #dict of dicts to store angle values for each node
            
        nt_a, nt_u, nt_g, nt_c = {},{},{},{}
        if(cpt<100):
            print(f'Reading {pdb_id}')
            cpt+=1
            # Load graph  
            g = pickle.load(open(os.path.join(gr_dir,pdb_id), 'rb'))
            nodes =g.nodes(data=True)
            N = g.number_of_nodes()
            
            # Clean edges
            remove_self_edges(g) # Get rid of self edges (not sure its right?)
            g=nx.to_undirected(g)
            
            # Add node features
            bad_nts = [] # nucleotides for which angles raise error 
            for n, data in g.nodes(data=True):
                nucleotide = data['nucleotide']
                
                # Nucleotide identity
                n_type = nucleotide.nt
                nt_a[n] = float(n_type=='A')
                nt_u[n] = float(n_type=='U')
                nt_g[n] = float(n_type=='G')
                nt_c[n] = float(n_type=='C')

                # Angles : 
                try:
                    chi, delta, gly_base = base_angles(nucleotide, 'rad')
                    d['chi'][n]=chi
                    d['delta'][n]=delta
                    d['gly_base'][n]= gly_base
                    
                except(IndexError): # missing atom in nucleotide
                    bad_nts.append(n)
                
            # Remove nodes 
            G = g.copy()
            G.remove_nodes_from(bad_nts)
            
            # Add node feature to all nodes 
            for a in angles:
                nx.set_node_attributes(G, d[a], a)
            # Nucleotide types
            nx.set_node_attributes(G, nt_a, 'A')
            nx.set_node_attributes(G, nt_u, 'U')
            nx.set_node_attributes(G, nt_g, 'G')
            nx.set_node_attributes(G, nt_c, 'C')
            
            # Save
            with open(os.path.join(annot_dir,pdb_id),'wb') as f:
                pickle.dump(G, f)
                
    print(f'wrote {cpt} preprocessed graphs to {args.write_dir}')
                
            
            