# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:50:14 2020

@author: jacqu

Load graphs with embeddings,
visualize node embeddings for occurences of 3D motifs : 
    
Compute fitting distance for motif occurrences, for different number of base differences. 
"""

import pickle
import networkx as nx 
import torch
import numpy as np

import os, sys
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(script_dir)
    sys.path.append('../data_processing')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--graphs_dir', help="path to training dataframe", type=str, default='../data/with_embeddings')
    
    ############
    
    args=parser.parse_args()
    
    # Params 
    data_dir = args.graphs_dir
    graphs = os.listdir(data_dir)
    emb_size = 32
    
    # Motifs data  
    with open('../data_processing/data_exploration/3dmotifs_dict.pickle','rb') as f:
        motifs_dict = pickle.load(f)
    
    with open('../data_processing/data_exploration/discrepancy_ranks_dict.pickle','rb') as f:
        ranks_dict = pickle.load(f)
        refs_dict = pickle.load(f)
    
    e_dict = {} # embeddings dict, keyed by motif
    g_dict = {} # pdb dict, keyed by motif name 
    refs_emb_dict = {} # reference embedding dict, keyed by motif. values are tuple (pdbid, embedding tensor)
    
    
    diffs={}
    
    for gid in graphs:
        
        embeddings = torch.zeros(40,emb_size)
        try:
            with open(os.path.join(data_dir,gid), 'rb') as f: # Load embedded graph 
                g = pickle.load(f)  
            with open(os.path.join('../data/motifs_chunks',gid),'rb') as f : # load motif name 
                _ = pickle.load(f)
                motif = pickle.load(f)
        except:
            continue
        
        index = 0
        nts = []
        for (n, data) in g.nodes(data=True): # Catch node embeddings 
            
            embeddings[index]=data['h']
            nt = data['nucleotide'].nt
            nts.append(nt)
            index+=1
            
        embeddings = embeddings[:index]
        
        tf = embeddings
        
        # Collect embedding 
        if(motif in e_dict):
            e_dict[motif].append(embeddings)
            g_dict[motif].append(nts)
        else:
            e_dict[motif]=[]
            e_dict[motif].append(embeddings)
            g_dict[motif]=[]
            g_dict[motif].append(nts)
            
        # Check if reference motif 
        if motif in refs_dict and refs_dict[motif]==gid[:4]:
            refs_emb_dict[motif] = (nts,embeddings)
            
    print('Built dictionary with motifs occurences')
    
    for motif in e_dict : 
        if(len(e_dict[motif])>1):
            
            print('motif :',motif)
            reduced_occ = e_dict[motif]
            
            n_nts = [a.shape[0] for a in reduced_occ]
            print('Number of nodes in each motif occurrence :', n_nts)
            # keep only occurrences with same nbr of nts
            
            # ref embedding :
            if(motif in refs_emb_dict):
                nts_ref , ref = refs_emb_dict[motif]
                ref = ref.numpy()

            else: 
                continue
            
            reduced_occ = [r for r in reduced_occ if r.shape[0]==len(nts_ref)]
            reduced_nts = [l for l in g_dict[motif] if len(l)==len(nts_ref)]
            
            if len(reduced_occ) <=1 : 
                continue
            
            for i in range(len(reduced_occ)):
                
                # for each valid occurrence of the motif 
                nts =reduced_nts[i]
                
                M = np.zeros((ref.shape[0], ref.shape[0]))
                emb_i = reduced_occ[i].numpy()
                fit_dist = 0
                diff = 0 
                for i in range(ref.shape[0]):
                    for j in range(ref.shape[0]):
                        M[i,j] = np.linalg.norm(emb_i[i]- ref[j])
                        
                    fit_dist += min(M[i,:])
                    match_for_i = np.argmin(M[i,:])
                    nt_match_for_i = nts_ref[match_for_i]
                    if(nts[i] != nt_match_for_i):
                        diff+=1
                fit_dist = fit_dist/len(nts_ref)
                diff = int(10*diff/len(nts_ref))
                if diff in diffs:
                    diffs[diff].append(fit_dist)
                else:
                    diffs[diff]=[fit_dist]
                
                print(fit_dist)
                print(nts_ref, nts)
                
                
for d in diffs :
    if(d<4):
        sns.distplot(diffs[d], label = str(d), hist = False, kde=True)
        plt.legend()
            
                
            