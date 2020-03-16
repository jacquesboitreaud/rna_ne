# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:30:04 2020

@author: jacqu
"""
import pickle
import networkx as nx
import sys, os 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(script_dir,'data_processing'))
    from data_processing.rna_classes import *
    
    
    with open('data/chunks/5i4a.pickle','rb') as f:
        g = pickle.load(f)
        
    print(g.nodes(data=True))
    
    for n, data in g.nodes(data=True):
        if(n==('B',11)):
            print(data)
    
    

