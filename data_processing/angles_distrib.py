# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:04:24 2020

@author: jacqu
"""

import pickle
import os

gr_dir = '../data/chunks'

graphs = os.listdir(gr_dir)

cpt=0 
graphs_cpt = 0
d = {'chi':[],
     'psi':[],
     'delta':[],
     'alpha':[],
     'beta':[],
     'gamma':[],
     'epsilon':[],
     'zeta':[]}

for gid in graphs:
    graphs_cpt+=1
    if(graphs_cpt%100==0):
        print(graphs_cpt)

    with open(os.path.join(gr_dir,gid),'rb') as f:
        g = pickle.load(f)
        nn = g.number_of_nodes()
        
    for n, data in g.nodes(data=True):
        # Get angles 
        alpha, beta, gamma, delta, epsilon, zeta, chi, gly_base = data['angles']
        d['chi'].append(chi)
        d['psi'].append(gly_base)
        d['delta'].append(delta)
        d['gamma'].append(gamma)
        d['beta'].append(beta)
        d['alpha'].append(alpha)
        d['zeta'].append(zeta)
        d['epsilon'].append(epsilon)
    
with open('../data/angles_distrib.pickle', 'wb') as f:
    pickle.dump(d,f)