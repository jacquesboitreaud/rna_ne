# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:09:30 2019

@author: jacqu

Utils functions for working with PDB files using biopython
"""

import numpy as np
import dgl
import torch


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g

def reordered(d, edgemap):
    # takes a dist matrix and list of edgetypes, reorders. 
    
    indexes, labels = [],[]
    
    # construct indexes 
    for e in ['S33', 'S35', 'S53', 'S55']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['B35']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['CWW', 'TWW']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['CHH', 'THH']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['CSS', 'TSS']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['CWS', 'CSW']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['TWS', 'TSW']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['CWH', 'CHW']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['TWH', 'THW']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['CSH', 'CHS']:
        indexes.append(edgemap[e])
        labels.append(e)
    for e in ['TSH', 'THS']:
        indexes.append(edgemap[e])
        labels.append(e)
        
    for e in edgemap.keys():
        if 'BPH' in e:
            labels.append(e)
            indexes.append(edgemap[e])
    for e in edgemap.keys():
        if 'BR' in e:
            labels.append(e)
            indexes.append(edgemap[e])
            
    labels.append('WAT')
    indexes.append(edgemap['WAT'])
    
    print(len(labels))
    
    print(indexes)
    
    d = d[indexes,:] # reorder 
    d = d[:,indexes] # reorder 
    
    
    return d, labels

def select_NC(d, edgemap):
    # takes a dist matrix and list of edgetypes, makes smaller matrix with only NC and WC edgetypes 
    
    indexes, labels = [],[]
    
    # construct indexes 
    for e in ['CHH', 'TWH', 'CWW','THS','CWS', 'CSS', 'CWH', 'CHS', 'TWS', 'TSS', 'TWW', 'THH']:
        indexes.append(edgemap[e])
        labels.append(e)
    
    print(len(labels))
    
    print(indexes)
    
    d = d[indexes,:] # reorder 
    d = d[:,indexes] # reorder 
    
    
    return d, labels
    


    