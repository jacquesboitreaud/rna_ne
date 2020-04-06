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

Preprocess graphs 
  
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
    from data_processing.angles import base_angles, norm_base_angles
    from data_processing.rna_classes import *
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing 'rna_classes' nx graphs ", 
                        type=str, default="C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs")
    
    parser.add_argument('-c', "--cutoff", help="Max number of train samples. Set to -1 for all graphs in dir", 
                        type=int, default=4500)
    
    parser.add_argument('-o', '--write_dir', help="path to directory to write preprocessed graphs ", 
                        type=str, default="../data/chunks")
    
    parser.add_argument('-hr', "--high_res", help="Use only high resolution PDB structures (406 samples).", 
                        type=bool, default=False)
    
    parser.add_argument('-m', "--motifs_only", help=" Parse only graphs with motifs ", 
                        type=bool, default=False)
    
    parser.add_argument('-mg', "--magnesium_only", help=" Parse only graphs with magnesium binding sites ", 
                        type=bool, default=True)
    
     # =======

    args=parser.parse_args()
    
    # Hyperparams 
    gr_dir = args.graphs_dir
    annot_dir = args.write_dir
    
    if(args.high_res):
        high_res_struc = pickle.load(open('data_exploration/high_res_pdb.pickle','rb'))
        selected = high_res_struc
        select = True
        print(f'>>> Parsing {len(selected)} high resolution structures')
        annot_dir = annot_dir + '_HR'

    if args.motifs_only:
        with_motifs = pickle.load(open('data_exploration/3dmotifs_dict.pickle','rb'))
        selected = with_motifs.keys()
        select = True
        print(f'>>> Parsing {len(selected)} graphs with motifs')
        annot_dir = "../data/motifs_graphs"
        
    if args.magnesium_only:
        mg_dict = pickle.load(open('../tasks/tasks_processing/mg_binding_dict.pickle','rb'))
        selected = mg_dict.keys()
        select = True
        print(f'>>> Parsing {len(selected)} graphs with magnesium binding sites')
        annot_dir = "../data/chunks_mg"
    
    nucleotides_id = {'A':0,
                      'U':1,
                      'G':2,
                      'C':3}
    
    print(f'>>> Calculating torsion angles and base normal vectors, for each nt.')
    print(f'Graphs with node features will be saved to {annot_dir}')
    
    cpt, bads =0,0
    nucleotides_counter = 0 
    # Load list of graphs to ignore
    #bad_graphs = set(pickle.load(open('bad_graphs.pickle','rb')))
    
    parse_dict = {}
        
    for pdb_id in os.listdir(gr_dir):
        
        if(cpt<args.cutoff):
            
            if ( select and (pdb_id[:4]  not in selected)):
                #print('ignoring graph')
                continue
            
            cpt+=1
            print(f'Reading {pdb_id}')
            # Dict for new node attributes 
            node_attrs = {}
            problem_nts = []
            
            node_attrs['angles']={} # dict to store angle values for each node
            node_attrs['base_norm_vec']={} # angles of base normal vectors 
            
            node_attrs['identity']={} # dict to store nucleotide identity for each node 
        
            # Load graph  
            g = pickle.load(open(os.path.join(gr_dir,pdb_id), 'rb'))
            
            # 1/ Remove dangling nodes from graph 
            
            nodes =g.nodes(data=True)
            N = g.number_of_nodes()
            
            # Clean edges
            remove_self_edges(g) # Get rid of self edges (not sure its right?)
            g=nx.to_undirected(g)
            g= dangle_trim(g)
            N1 = g.number_of_nodes()
            
            # ================ Nucleotides parsing ===================
            for n, data in g.nodes(data=True):
                nucleotide = data['nucleotide']
                pdb_pos = int(nucleotide.pdb_pos)
                
                # Count context nodes
                nbr_neigh = len(g[n])
                
                if(nbr_neigh==0):
                    problem_nts.append(n)
                else:
                    
                    # Prev and next nucleotides 
                    prev_nt = find_nucleotide(g,n[0], pdb_pos-1)
                    next_nt = find_nucleotide(g,n[0], pdb_pos+1)
                    
                    # Nucleotide identity
                    n_type = nucleotide.nt
                    a = [0,0,0,0]
                    try:
                        a[nucleotides_id[n_type]]=1
                    except: # if weird nucleotide, a is set to all zeros
                        pass
                    node_attrs['identity'][n]=a
    
                    # Angles : 
                    angles = base_angles(nucleotide, prev_nt, next_nt)
                    base_normal_vec = norm_base_angles(nucleotide)
                    nonzero = np.count_nonzero(angles)
                    if(nonzero<8): # Missing angles 
                        problem_nts.append(n) 
                        
                    # Store in node attributes dict 
                    node_attrs['angles'][n]=angles
                    node_attrs['base_norm_vec'][n]=base_normal_vec
            
            # ========= Create features and check all angles !=0 =============
        
            # Remove lonely nucleotides 
            G = g.copy()
            G.remove_nodes_from(problem_nts)
            
            # check number of nodes AND all nodes have at least one neighbor:
            
            N1 = G.number_of_nodes()
            if(N1<8): # Not enough nodes, do not process and do not save 
                print('less than 8 nodes. passing')
                bads+=1
                continue # empty graph, do not process and do not save 


            lonely_nodes = [n for n in G.nodes() if len(G[n])==0]
            G.remove_nodes_from(lonely_nodes)
            N1 = G.number_of_nodes()
            if(N1<8): # Do not save this graph 
                print('less than 8 nodes. passing')
                bads+=1
                continue
            
            nucleotides_counter += N1
            
            # Add node feature to all nodes 
            assert(len(node_attrs['angles']) >= G.number_of_nodes())
            nx.set_node_attributes(G, node_attrs['angles'], 'angles')
            nx.set_node_attributes(G, node_attrs['identity'], 'identity')
            
            
            # Save
            with open(os.path.join(annot_dir,pdb_id),'wb') as f:
                pickle.dump(G, f)
                
    print(f'wrote {cpt} preprocessed graphs to {annot_dir}')
    print(f'removed {bads} too small graphs')
    print(f'Parsed {nucleotides_counter} clean nucleotides')  
            
            