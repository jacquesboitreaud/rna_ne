# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Prepare graphs for downstream task:
Annotates graphs with magnesium binding sites, and saves all graphs containing at least a binding site to new directory

        
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
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))
    sys.path.append(os.path.join(script_dir, '../..'))

    from pdb_utils import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing 'rna_classes' nx graphs ", 
                        type=str, default="../../data/chunks")
    parser.add_argument('-o', '--out_dir', help="path to directory to save mg_graphs ", 
                        type=str, default="../../data/mg_graphs")
    parser.add_argument('-c', "--cutoff", help="Distance cutoff for binding sites, in Angstrom", 
                        type=int, default=6)
    
    parser.add_argument('-d', "--binding_sites_dict", help="Path to binding sites dictionary", 
                        type=str, default='C:/Users/jacqu/Documents/MegaSync Downloads/new_mg_res.p')
    
    # =======

    args=parser.parse_args()
    
    # Open dict of magnesium binding res
    with open(args.binding_sites_dict, 'rb') as f:
        sites = pickle.load(f)
    print(f'Loaded binding sites dictionary for {len(sites)} PDB structures')
    cpt, with_site =0, 0
    
    # Parse graphs     
    for pickle_id in os.listdir(args.graphs_dir):
        pdbid = pickle_id[:-7]
        cpt+=1
        
        # Load graph  
        g = pickle.load(open(os.path.join(args.graphs_dir,pickle_id), 'rb'))
        print(f'Parsing {pdbid}')
        
        try:
            g_sites = sites[pdbid+'.cif']
            n_sites = len(g_sites)
        except(KeyError):
            print('no binding site found')
            continue
        
        # initialize dict for new node feature : binds mg 
        node_feat = {}
        
        for n, data in g.nodes(data=True):
            chain, idx = n 
            base = data['nucleotide'].nt
            
            res_id=f'{chain}:{base}:{idx}'
            
            is_binding = 0
            counter = 0
            for s in g_sites: # for each binding site found in g 
                # s = ( ligand_name, dict of residues ) 
                if(res_id in s[1][args.cutoff]['rna_res']): 
                    # Residue is in binding site at cutoff distance 
                    is_binding = 1 
                    counter +=1
            # Append to node features dict 
            node_feat[n]=is_binding
        print(f'{counter} binding site nucleotides annotated, {len(s[1][args.cutoff]["rna_res"])} in dict')
        print(s[1][args.cutoff]['rna_res'])
            
        # Add node feature to nx graph 
        nx.set_node_attributes(g,node_feat, name = 'Mg_binding')
        
        with open(os.path.join(args.out_dir,pickle_id), 'wb') as f:
            pickle.dump(g,f)
            
        with_site+=1 
        
    print(f'Job finished, parsed {cpt} PDBs, {with_site} have binding site(s) and were saved')
            
            
            