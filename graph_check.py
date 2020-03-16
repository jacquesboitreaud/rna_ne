# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:30:04 2020

@author: jacqu
"""
import pickle
import networkx as nx

with open('data/chunks/5i4a.pickle','rb') as f:
    g = pickle.load(f)
    
print(g.nodes(data=True))
    
    

