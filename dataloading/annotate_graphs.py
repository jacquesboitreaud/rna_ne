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
from dataloading.graph_process import *

# Hyperparams 
gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
annot_dir = "C:/Users/jacqu/Documents/GitHub/data/annotated"
sup = SVDSuperimposer()
cpt=0

for pdb_id in os.listdir(gr_dir):
    # Dict for RMSD
    distances={}
    ## DEBUG
    if(cpt<10):
        print(f'Reading {pdb_id}')
        cpt+=1
        # Load graph  
        g = pickle.load(open(os.path.join(gr_dir,pdb_id), 'rb'))
        nodes =g.nodes(data=True)
        N = g.number_of_nodes()
        
        # Clean edges before using edges list for RMSDs
        remove_self_edges(g) # Get rid of self edges (not sure its right?)
        g=nx.to_undirected(g)
        
        # Iterate over NON BACKBONE graph edges: 
        for n_a, n_b ,label in g.edges(data=True):
            if(label not in ['B35','B53'] ):
                nt_a, nt_b = nodes[n_a]['nucleotide'], nodes[n_b]['nucleotide']
                pos_a, pos_b = int(nt_a.pos), int(nt_b.pos)
                
                # Get neighbors of rank 1 in chain's backbone
                neighbors_a = [nodes[n]['nucleotide'] for n in g.nodes() if abs(n[1]-pos_a)<=1]
                neighbors_b = [nodes[n]['nucleotide'] for n in g.nodes() if abs(n[1]-pos_b)<=1]
                
                # If one node has only one neighbor of rank 1: remove and compare at rank 0
                if(len(neighbors_a)!=len(neighbors_b)):
                    neighbors_a, neighbors_b = [nt_a], [nt_b]
                
                # Get the atoms (list of lists of atoms)
                atoms_a = [n.atoms for n in neighbors_a]
                atoms_b = [n.atoms for n in neighbors_b]
                
                coords_a=[]
                for atoms in atoms_a:
                    # Get key atoms coordinates 
                    for a in atoms:
                        if(a.atom_label in ['N1','C2','N3','C3','C4','N5','C5','C6']):
                            # Get coordinates 
                            coords_a.append((float(a.x),float(a.y), float(a.z)))
                coords_a=np.array(coords_a)
                
                coords_b=[]
                for atoms in atoms_b:
                    # Get key atoms coordinates 
                    for a in atoms:
                        if(a.atom_label in ['N1','C2','N3','C3','C4','N5','C5','C6']):
                            # Get coordinates 
                            coords_b.append((float(a.x),float(a.y), float(a.z)))
                coords_b=np.array(coords_b)
                
                if (coords_a.shape==coords_b.shape and coords_a.shape[0]>0):
                    rmsd = compute_rmsd(sup, coords_a, coords_b)
                    # Add node pair to dataset
                    distances[(n_a,n_b)]=rmsd
                    # DEBUG : STOP AFTER ONE COMPUTATION
                    #sys.exit()
                else:
                    print("Neighborhoods have different n° of nucleotides. Cannot compute rmsd")
                
            
        # Dump graph, distances_dict to pickle file 
        with open(os.path.join(annot_dir,pdb_id),'wb') as f:
            pickle.dump((g,distances), f)
            
            
            