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
import itertools
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

from random import random

LOCAL=False #server or local

# Paths
if(LOCAL):
    gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
    savedir = "C:/Users/jacqu/Documents/GitHub/data/DeepFRED_data"
    pdb_dir = "C:/Users/jacqu/Documents/databases/rcsb_pdb"
else:
    gr_dir="../../data/RNA_Graphs"
    savedir = '../../data/DeepFRED_data'
    pdb_dir = '../../data/rcsb_pdb'

if(__name__=='__main__'):
    
    sup = SVDSuperimposer()
    IO_writer = PDBIO()
    cpt=0 # PDB counter
    chunks_counter = 0 
    data_dict={}
    TMS=[]
    
    k=2 # Number of hops allowed to be counted in neighborhood
    
    faces = ['W', 'S', 'H']
    orientations = ['C', 'T']
    valid_edges = set([orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations])
    valid_edges.remove('CWW')
    
    for pickle_id in os.listdir(gr_dir):
        pdbid = pickle_id[:-7]
        # Dict for RMSD
        distances={}
        print(f'Reading {pdbid}')
        # Load graph  
        g = pickle.load(open(os.path.join(gr_dir,pickle_id), 'rb'))
        
        nodes =g.nodes(data=True)
        N = g.number_of_nodes()
        remove_self_edges(g) # Get rid of self edges (not sure it works?)
        g=nx.to_undirected(g)
        
        nodepair_counter=0
        
        # Load PDB file:
        pdbpath = os.path.join(pdb_dir,f'{pdbid}.cif',f'{pdbid}.cif')
        try:
            structure = read_pdb(pdbid, pdbpath)
            IO_writer.set_structure(structure)
            
            # Find non backbone graph edges 
            iter_edges = [e for e in g.edges(data=True) if e[2]['label'] not in ['B35','B53','S35','S53','S55','S33','CWW']]
            
            # Iterate over NON BACKBONE graph edges: 
            for n_a, n_b ,label in iter_edges:
                    nt_a, nt_b = nodes[n_a]['nucleotide'], nodes[n_b]['nucleotide']
                    pos_a, pos_b = int(nt_a.pos), int(nt_b.pos)
                    nodepair_counter+=1
                    
                    # get k-hops reachable nodes for na and nb
                    khops_a = nx.single_source_shortest_path_length(g, n_a, cutoff=k)
                    khops_b = nx.single_source_shortest_path_length(g, n_b, cutoff=k)
                    
                    #Get pdb position of n_a neighbors
                    pdb_1 = [int(nodes[n]['nucleotide'].pdb_pos) for n in khops_a]
                    pdb_1 += [int(nodes[n]['nucleotide'].pdb_pos) for n in khops_b]
                    
                    IO_writer.save('tmp/pdb_1.pdb', selectResidues(pdb_1))
                    
                    #Iterate over edges again : 
                    for n_c, n_d ,label in iter_edges:
                        if(random()>0.5 or (n_c==n_a and n_d==n_b)):
                            next
                        else:
                            nt_c, nt_d = nodes[n_c]['nucleotide'], nodes[n_d]['nucleotide']
                            pos_c, pos_d = int(nt_c.pos), int(nt_d.pos)
                            nodepair_counter+=1
                            
                            # get k-hops reachable nodes for na and nb
                            khops_c = nx.single_source_shortest_path_length(g, n_c, cutoff=k)
                            khops_d = nx.single_source_shortest_path_length(g, n_d, cutoff=k)
                            graph_chunk = g.subgraph({**khops_a , **khops_b,
                                                      **khops_c , **khops_d})
                            
                            #Get pdb position of n_a neighbors
                            pdb_2 = [int(nodes[n]['nucleotide'].pdb_pos) for n in khops_c]
                            pdb_2 += [int(nodes[n]['nucleotide'].pdb_pos) for n in khops_d]
                            
                            
                            IO_writer.save('tmp/pdb_2.pdb', selectResidues(pdb_2))
                                
                        
                        
                            with open("tmp/align.out", "w") as rnaout:
                                p = subprocess.run(["/home/mcb/users/jboitr/RNAalign/RNAalign", 
                                                    "tmp/pdb_1.pdb", "tmp/pdb_2.pdb"], stdout=rnaout)
                        
                            #get the output tmscore
                            tmscore = get_score("tmp/align.out")
                            if(tmscore<0): # Error happened 
                                next
                            else:
                                #Save edges, subgraph and tmscore
                                filename=pdbid+'_'+str(nodepair_counter)+'.pickle'
                                with open(os.path.join(savedir,filename),'wb') as f:
                                    pickle.dump(graph_chunk,f)
                                    pickle.dump([(n_a, n_b),(n_c,n_d), tmscore],f)
                                    #print(tmscore)
                                    TMS.append(tmscore)
                                    
            # If the structure was successfully processed
            cpt+=1  
            data_dict[pdbid]=nodepair_counter
            chunks_counter+= nodepair_counter # All chunks counter 
            np.save('final_dist.npy',TMS)
                                
                                
                        
        except(FileNotFoundError):
            next
    print('Job finished, number of PDBs processed : ', cpt)
    np.save('data_dict.npy', data_dict)
            
            
            