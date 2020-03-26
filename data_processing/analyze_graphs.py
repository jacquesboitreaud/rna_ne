# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:11:42 2019

@author: jacqu

Reads off-the-shelf RNA graphs (structure using rna_classes.py)
Preprocesses : 
    removes dangling nodes 
    Checks graph not empty
    computes 3D node features (base angles)
    adds nucleotide identity as a node feature

Parses graphs and counts nucleotides with missing coordinates and dangling nucleotides 
  
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
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing 'rna_classes' nx graphs ", 
                        type=str, default="C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs")
    parser.add_argument('-c', "--cutoff", help="Max number of train samples. Set to -1 for all graphs in dir", 
                        type=int, default=4)
    
    parser.add_argument('-o', '--write_dir', help="path to directory to write preprocessed graphs ", 
                        type=str, default="../data/chunks")
    
    parser.add_argument('-d', "--debug", help="debug", 
                        type=bool, default=True)
    
     # =======

    args=parser.parse_args()
    
    # Hyperparams 
    gr_dir = args.graphs_dir
    annot_dir = args.write_dir
    
    angles = ['alpha', 'beta', 'gamma', 'chi', 'delta', 'gly_base']
    
    print(f'Calculating {len(angles)} angles for each nt.')
    print(f'Graphs with node features will be saved to {annot_dir}')
    
    cpt, bads =0,0
    
    parse_dict = {}
        
    
    for pdb_id in os.listdir(gr_dir):
        
        if(cpt<args.cutoff):
            cpt+=1
            print(f'Reading {pdb_id}')
        
            # Add key in parse dict to store information about this graph
            parse_dict[pdb_id]={}
            lonely_nts=[]
            weird_nt_type = 0 
            no_prev = 0
            no_next=0
            no_prev_next = 0
            atoms_error = 0 
            perfect = 0 
            
            # Dict for new node attributes 
            node_attrs = {}
            for a in angles:
                node_attrs[a]={} #dict of dicts to store angle values for each node
                
            nt_a, nt_u, nt_g, nt_c = {},{},{},{}
        
            # Load graph  
            g = pickle.load(open(os.path.join(gr_dir,pdb_id), 'rb'))
            
            # 1/ Remove dangling nodes from graph 
            
            nodes =g.nodes(data=True)
            N = g.number_of_nodes()
            parse_dict[pdb_id]['num_nodes_init']=N
            
            # Clean edges
            remove_self_edges(g) # Get rid of self edges (not sure its right?)
            g=nx.to_undirected(g)
            g= dangle_trim(g)
            N1 = g.number_of_nodes()
            if(N1==0):
                continue # empty graph, do not process and do not save 
            
            # ================ Nucleotides parsing ===================
            for n, data in g.nodes(data=True):
                nucleotide = data['nucleotide']
                pdb_pos = int(nucleotide.pdb_pos)
                
                # Count context nodes
                nbr_neigh = len(g[n])
                
                if(nbr_neigh==0):
                    lonely_nts.append(n)
                else:
                    
                    # Prev and next nucleotides 
                    prev_nt = find_nucleotide(g,n[0], pdb_pos-1)
                    next_nt = find_nucleotide(g,n[0], pdb_pos+1)
                    if (prev_nt==None and next_nt==None):
                        no_prev_next +=1
                    elif prev_nt==None:
                        no_prev +=1
                    elif next_nt==None:
                        no_next +=1 
                    
                    # Nucleotide identity
                    n_type = nucleotide.nt
                    nt_a[n] = float(n_type=='A')
                    nt_u[n] = float(n_type=='U')
                    nt_g[n] = float(n_type=='G')
                    nt_c[n] = float(n_type=='C')
                    if(n_type not in {'A','U','G','C'}):
                        weird_nt_type +=1
    
                    # Angles : 
                    angles = base_angles(nucleotide, prev_nt, next_nt)
                    nonzero = np.count_nonzero(angles)
                    if(nonzero<5): # one angle that doesnot require prev or next failed 
                        atoms_error +=1
                    if(nonzero==8):
                        perfect +=1 
                        
                    # Store in node attributes dict 
                    #TODO
                    #node_attrs['chi'][n]=angles[-2]
                        
            # ==========================================================
               
            # add info to dict 
            if args.debug:
                parse_dict[pdb_id]['lonely nts']=len(lonely_nts)
                parse_dict[pdb_id]['no prev']=no_prev
                parse_dict[pdb_id]['no next']=no_next
                parse_dict[pdb_id]['no prev/next']=no_prev_next
                parse_dict[pdb_id]['atoms pb']=atoms_error
                parse_dict[pdb_id]['weird_nt_type']=weird_nt_type
                parse_dict[pdb_id]['perfect']=perfect
                
                continue
            
            # ========= Create features and check all angles !=0 =============
        
            # Remove lonely nucleotides 
            G = g.copy()
            G.remove_nodes_from(lonely_nts)
            
            # check all nodes have at least one neighbor:
            
            nbr_neigh = [len(G[n]) for n in G.nodes()]
            m = min(nbr_neigh)
            if(m==0): # Do not save this graph : one node is lonely . 
                print('Lonely node(s). passing')
                bads+=1
                continue
            
            N1 = G.number_of_nodes()
            if(N1<=4): # Not enough nodes, do not process and do not save 
                print('less than 4 nodes. passing')
                bads+=1
                continue # empty graph, do not process and do not save 
            
            # Add node feature to all nodes 
            for a in angles:
                assert(len(node_attrs[a]) == G.number_of_nodes())
                nx.set_node_attributes(G, node_attrs[a], a)
            # Nucleotide types
            nx.set_node_attributes(G, nt_a, 'A')
            nx.set_node_attributes(G, nt_u, 'U')
            nx.set_node_attributes(G, nt_g, 'G')
            nx.set_node_attributes(G, nt_c, 'C')
            
            # Save
            cpt+=1
            with open(os.path.join(annot_dir,pdb_id),'wb') as f:
                pickle.dump(G, f)
                
    # Save debug dict 
    if(args.debug):
        print('writing pdb parse dict')
        with open('pdb_parsing_dict.pickle','wb') as f:
                pickle.dump(parse_dict, f)
                
    print(f'wrote {cpt} preprocessed graphs to {annot_dir}')
    print(f'removed {bads} too small graphs')
                
            
            