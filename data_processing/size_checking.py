# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Cut graphs into chunks and saves as pickle files.

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
    
    parser.add_argument('-i', '--graphs_dir', help="path to directory containing already processed graphs", 
                        type=str, default="../data/chunks")
    
     # =======

    args=parser.parse_args()
    
    # Hyperparams 
    gr_dir = args.graphs_dir
    
    large = []
    cpt=0
    
    for pdb_id in os.listdir(gr_dir):
        cpt+=1
        if(cpt%100)==0:
            print(cpt)
         
        # Load graph  
        g = pickle.load(open(os.path.join(gr_dir,pdb_id), 'rb'))
        
        # 1/ Remove dangling nodes from graph 
        
        nodes =g.nodes(data=True)
        N = g.number_of_nodes()
         
        if(N>5000 or N < 8):
            print('too large / too small graph , ', N)
            large.append(pdb_id)
             
             
with open('large_graphs.pickle','wb') as f:
    pickle.dump(large,f)
