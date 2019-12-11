# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:54:59 2019

@author: jacqu
"""

import pickle 
import networkx as nx 
from rna_classes import *


g = pickle.load(open('../../data/DeepFRED_data/2uuc_1352.pickle','rb'))

iter_edges = [(e[2]['label']) for e in g.edges(data=True)]

