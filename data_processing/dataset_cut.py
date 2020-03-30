# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:47:13 2020

@author: jacqu
"""

with open('selected_nodes.pickle','rb') as f:
    d = pickle.load(f)
    
keys = list(d.keys())

train, valid = keys[:-10], keys[-10:]



valid_nodes = {k:d[k] for k in valid  }

for k in valid : 
    d.pop(k)
    
with open('train_nodes.pickle','wb') as f:
    pickle.dump(d,f)
    
with open('valid_nodes.pickle','wb') as f:
    pickle.dump(valid_nodes,f)    