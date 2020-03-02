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
from rna_classes import *
from Bio.SVDSuperimposer import SVDSuperimposer
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

    from dataloading.pdb_utils import *
    from dataloading.angles import chi_angle

    # Hyperparams 
    gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
    annot_dir = "../data/chunks"

    cpt=0
    
    for pdb_id in os.listdir(gr_dir):
        
        # Dict for new node attributes 
        chi_torsions={}
        ## DEBUG
        if(cpt<10):
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
            for n, data in g.nodes(data=True):
                nucleotide = data['nucleotide']
                
                chi = chi_angle(nucleotide, 'deg')
                chi_torsions[n]=chi
                
                #TODO: add other angles if necessary 
            
            # Add node feature to all nodes 
            nx.set_node_attributes(g, chi_torsions, 'chi')
            
            # Cut graph into chunks 
            
            # Save chunks 
                    
            with open(os.path.join(annot_dir,pdb_id),'wb') as f:
                pickle.dump(g, f)
                
            
            