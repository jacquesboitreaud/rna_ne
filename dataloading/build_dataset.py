# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Reads RNA graphs annotated with FR3D edge labels 

For each interesting pair of nodes (non backbone edges):
    - cuts the graph to the neighborhood of interest
    
    - computes the RMSD in 3D 
    
    - saves triplet ((n1,n2) subgraph, rmsd) to pickle file 
    
DEBUG: on 6n2v pdb (contains only RNA and ligands)

        
"""

import numpy as np
import pickle 
import os 
import networkx as nx
from rna_classes import *
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBIO
import sys
import subprocess
if __name__ == "__main__":
    sys.path.append("..")

from pdb_utils import *
from utils import *

# Hyperparams 
gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
annot_dir = "C:/Users/jacqu/Documents/GitHub/data/annotated"
savedir = "C:/Users/jacqu/Documents/GitHub/data/DeepFRED_data"

if(__name__=='__main__'):
    
    sup = SVDSuperimposer()
    IO_writer = PDBIO()
    cpt=0
    
    k=5 # Number of hops allowed to be counted in neighborhood
    
    for pickle_id in os.listdir(gr_dir):
        pdbid = pickle_id[:-7]
        # Dict for RMSD
        distances={}
        ## DEBUG
        if(cpt<1):
            print(f'Reading {pdbid}')
            cpt+=1
            # Load graph  
            g = pickle.load(open(os.path.join(gr_dir,pickle_id), 'rb'))
            
            nodes =g.nodes(data=True)
            N = g.number_of_nodes()
            remove_self_edges(g) # Get rid of self edges (not sure it works?)
            g=nx.to_undirected(g)
            
            nodepair_counter=0
            
            # Load PDB file 
            structure = read_pdb(pdbid)
            IO_writer.set_structure(structure)
            
            
            # Iterate over NON BACKBONE graph edges: 
            for n_a, n_b ,label in g.edges(data=True):
                if(label not in ['B35','B53'] ):
                    nt_a, nt_b = nodes[n_a]['nucleotide'], nodes[n_b]['nucleotide']
                    pos_a, pos_b = int(nt_a.pos), int(nt_b.pos)
                    nodepair_counter+=1
                    
                    # get k-hops reachable nodes for na and nb
                    khops_a = nx.single_source_shortest_path_length(g, n_a, cutoff=k)
                    khops_b = nx.single_source_shortest_path_length(g, n_a, cutoff=k)
                    graph_chunk = g.subgraph({**khops_a , **khops_b})
                    
                    #TODO: Get PDB neighborhoods to align , save two small pdbs
                    #Get pdb position of n_a neighbors
                    pdb_a = [int(nodes[n]['nucleotide'].pdb_pos) for n in khops_a]
                    pdb_b = [int(nodes[n]['nucleotide'].pdb_pos) for n in khops_b]
                    
                    
                    IO_writer.save('../tmp/pdb_a.pdb', selectResidues(pdb_a))
                    IO_writer.save('../tmp/pdb_b.pdb', selectResidues(pdb_b))
                    
                    #TODO: run RNA-align and compute tmscore
                    ######
                    subprocess.call(["/home/mcb/users/jboitr/RNAalign/RNAalign", "tmp/pdb_a.pdb", "tmp/pdb_b.pdb"])
                    tmscore = 1 
                    
    
                    #Save n_a, n_b, subgraph and tmscore
                    filename=pdbid+'_'+str(nodepair_counter)+'.pickle'
                    with open(os.path.join(savedir,pdbid),'wb'):
                        pickle.dump(graph_chunk)
                        pickle.dump((n_a, n_b, tmscore))
            
            
            