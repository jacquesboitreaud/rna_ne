# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Prepare graphs for downstream task:
counts number of MG binding residues overall and creates a dict of mg binding residues per graph 
        
"""

import numpy as np

import pickle 
import os 
import networkx as nx
import sys
import argparse


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.join(script_dir,os.pardir, os.pardir)
    sys.path.append(os.path.join(repo_root, 'data_processing'))

    from rna_classes import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing annotated nx graphs ", 
                        type=str, default=os.path.join(repo_root,"../../MEGAsync Downloads/mg_graphs"))
    
    
    # =======

    args=parser.parse_args()
    
    # Open dict of magnesium binding res

    cpt, total_mg =0, 0
    total_res = 0
    chem_modif =0
    
    d = {}
    
    # Parse graphs     
    for pickle_id in os.listdir(args.graphs_dir):
        pdbid = pickle_id[:-3]
        cpt+=1
        
        d[pdbid] = set()
        
        # Load graph  
        g = pickle.load(open(os.path.join(args.graphs_dir,pickle_id), 'rb'))
        print(f'Parsing {pdbid}')
        total_res += g.number_of_nodes()
        
        # initialize dict for new node feature : binds mg 
        node_feat = {}
        counter_found = 0 # nbr nodes found in dict 
        
        for n, data in g.nodes(data=True):
            chain, _ = n 
            pos = data['pdb_pos']
            
            res_id=f'{chain}:{pos}'
            
            if(data['mg']):
                counter_found+=1
                total_mg+=1
                identifier = f'{chain}:{pos}'
                d[pdbid].add(identifier)
                
                if(data['chemically_modified']):
                    chem_modif +=1 
                
            
        print(pdbid, counter_found, ' mg binding residues')
        
    print(f'Job finished, parsed {cpt} PDBs, {total_mg} binding residues over {total_res}')
    print(chem_modif, ' chemically modified binding residues')
    
    bads=[]
    for k in d.keys():
        if(len(d[k])==0):
            bads.append(k)
    for k in bads:
        del(d[k])
    print(f'{len(d)} graphs with MG binding nucleotides')
    
    with open('mg_binding_dict.pickle','wb') as f:
        pickle.dump(d,f)
            
            
            