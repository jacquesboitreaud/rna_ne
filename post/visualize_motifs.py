# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:50:14 2020

@author: jacqu

Load graphs with embeddings,
visualize node embeddings for occurences of 3D motifs : 
    
The goal is to see if we can see isomorphic nodes in motif occurences by just looking at their embeddings 
    
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
        for (n, data) in g.nodes(data=True): # Catch node embeddings 
            
            embeddings[index]=data['h']
            index+=1
            
        embeddings = embeddings[:index]
        
        tf = embeddings
        
        # Collect embedding 
        if(motif in e_dict):
            e_dict[motif].append(embeddings)
            g_dict[motif].append(gid[:-7])
        else:
            e_dict[motif]=[]
            e_dict[motif].append(embeddings)
            g_dict[motif]=[]
            g_dict[motif].append(gid[:-7])
            
        # Check if reference motif 
        if motif in refs_dict and refs_dict[motif]==gid[:4]:
            refs_emb_dict[motif] = (gid,embeddings)
            
    print('Built dictionary with motifs occurences')
    
    for motif in e_dict : 
        if(len(e_dict[motif])>1):
            
            print('motif :',motif)
            occurrences = e_dict[motif]
            
            n_nts = [a.shape[0] for a in occurrences]
            print('Number of nodes in each motif occurrence :', n_nts)
            # keep only occurrences with same nbr of nts
            max_nodes = max(n_nts)
            reduced_occ = [a for a in occurrences if a.shape[0]==max_nodes] 
            print('Number of full occurrences :', len(reduced_occ))
            
            # ref embedding :
            if(motif in refs_emb_dict):
                ref_id, ref = refs_emb_dict[motif]
                ref = ref.numpy()
                ref_id=ref_id[:-7]
            else: 
                continue
            
            if len(reduced_occ) <=1 : 
                continue
            
            """
            # =================================================================
            #For visualization : 
            #fit pca to all motif occurences 
            t = torch.cat([l for l in reduced_occ])
            t= np.array(t)
            pca = PCA()
            pca.fit(t)
            
            plt.figure()
            colors = sns.color_palette()
            
            for i in range(min(len(reduced_occ),8)):
                tf = pca.transform(reduced_occ[i])
                sns.scatterplot(tf[:,0], tf[:,1], c=colors[i], label=f'{i}')
            plt.title(f'{m}')
            # =================================================================
            """
            
            for i in range(len(reduced_occ)):
                
                # for each valid occurrence of the motif 
                id_i = g_dict[motif][i][:4]
                rank_i = ranks_dict[(id_i, motif)]
                
                M = np.zeros((max_nodes, ref.shape[0]))
                emb_i = reduced_occ[i].numpy()
                fit_dist = 0
                for i in range(max_nodes):
                    for j in range(ref.shape[0]):
                        M[i,j] = np.linalg.norm(emb_i[i]- ref[j])
                    fit_dist += min(M[i,:])
                
                
                
                print(motif, ref_id, id_i , rank_i, fit_dist/max_nodes)
