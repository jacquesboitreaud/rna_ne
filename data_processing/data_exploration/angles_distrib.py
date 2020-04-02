# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:04:24 2020

@author: jacqu

Retrieve angles distributions for high-resolution structures (2A or better for now)
"""

import pickle
import os
import sys
sys.path.append('..')

gr_dir = '../../data/chunks'
high_res_list = pickle.load(open('high_res_pdb.pickle', 'rb'))

graphs = os.listdir(gr_dir)

cpt=0 
graphs_cpt = 0
d = {'nt':[],
     'chi':[],
     'psi':[],
     'delta':[],
     'alpha':[],
     'beta':[],
     'gamma':[],
     'epsilon':[],
     'zeta':[]}

for gid in graphs:
    if(gid[:4] in high_res_list):
        graphs_cpt+=1
        
        if(graphs_cpt%10==0):
            print(graphs_cpt)
    
        with open(os.path.join(gr_dir,gid),'rb') as f:
            g = pickle.load(f)
            nn = g.number_of_nodes()
            
        for n, data in g.nodes(data=True):
            # Get angles 
            alpha, beta, gamma, delta, epsilon, zeta, chi, gly_base = data['angles']
            d['nt'].append(data['nucleotide'].nt)
            d['chi'].append(chi)
            d['psi'].append(gly_base)
            d['delta'].append(delta)
            d['gamma'].append(gamma)
            d['beta'].append(beta)
            d['alpha'].append(alpha)
            d['zeta'].append(zeta)
            d['epsilon'].append(epsilon)
            
print(f'{graphs_cpt} graphs with high resolution were found over 406. ')
    
with open('angles_distrib_HR.pickle', 'wb') as f:
    pickle.dump(d,f)