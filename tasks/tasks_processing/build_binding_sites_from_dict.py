# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Prepare graphs for downstream task:

    
Samples binding sites and non binding sites chunks from RNA ggraphs_dir
"""

import numpy as np

import pickle 
import os 
import networkx as nx
import sys
import argparse
import random


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.join(script_dir,os.pardir, os.pardir)
    sys.path.append(os.path.join(repo_root, 'data_processing'))

    from rna_classes import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing 'rna_classes' nx graphs ", 
                        type=str, default="../data/mg_annotated_graphs")
    
    parser.add_argument('-o', '--out_dir', help="path to directory to save mg_graphs ", 
                        type=str, default=os.path.join(script_dir,"../data/mg_sites"))
    
    parser.add_argument('-d', "--binding_sites_dict", help="Path to binding sites dictionary", 
                        type=str, default=os.path.join(script_dir,"mg_binding_dict.pickle"))
    
    # =======

    args=parser.parse_args()
    
    # Open dict of magnesium binding res
    with open(args.binding_sites_dict, 'rb') as f:
        sites = pickle.load(f)
    print(f'********** Loaded binding sites dictionary with {len(sites)} PDB structures*********')
    cpt, with_site =0, 0
    
    for k,v in sites.items():
        print(k, len(v))
        
    s = sorted(sites.keys())
    
    done = s[:216] 
    
    # Parse graphs     
    for pdbid in sites.keys():
        pickle_id = pdbid+'.pickle'
        cpt+=1
        
        if pdbid in done :
            continue
        
        # Load graph  
        try:
            g = pickle.load(open(os.path.join(args.graphs_dir,pickle_id), 'rb'))
            g_sites = sites[pdbid]
            n_sites = len(g_sites)
        except:
            print(f'{pdbid}: not found')
            continue
        
        # initialize dict for new node feature : binds mg 
        node_feat = {}
        counter_found = 0 # nbr nodes found in dict 
        counter_samples = 0
        
        # Collect Mg binding nucleotides
        ntot = len(g_sites)
        print('num nodes : ', g.number_of_nodes())
        p = ntot/g.number_of_nodes()
            
        for n, data in g.nodes(data=True):
            chain, _ = n 
            pos = data['nucleotide'].pdb_pos
            
            res_id=f'{chain}:{pos}'
            r= np.random.rand()
            
            if(res_id in g_sites or r<p): # Binding sites to build from 
            
                sub = nx.ego_graph(g, n, radius=1, undirected = True)
                
                mg = nx.get_node_attributes(sub, 'Mg_binding')
                
                is_binding = max(mg.values())
                
                if is_binding>0: 
                    # Residue is in binding site at cutoff distance 
                    counter_found +=1
                    
                    with open(os.path.join(args.out_dir,f'{pdbid}_{str(counter_samples)}'), 'wb') as f:
                        pickle.dump((sub,1),f)
                    counter_samples+=1
                
                else: # non site , keep with proba 1/3
                    with open(os.path.join(args.out_dir,f'{pdbid}_{str(counter_samples)}'), 'wb') as f:
                        pickle.dump((sub,0),f)
                    counter_samples+=1
                    
        print(f'{pdbid}: {counter_found} binding site saved, {counter_samples} total, {ntot} in dict')
        
    print(f'Job finished, parsed {cpt} PDBs, {with_site} have binding site(s) and were saved')
            
            
            