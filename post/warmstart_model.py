# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019
@author: jacqu

"""
#  Loading the pretrain embeddings GNN : 
from model import Model

init_embeddings = Model(features_dim = 12, h_dim = 16, out_dim = 32, num_rels = 44, radii_params=(1,1,2),
                   num_bases =10)
init_embeddings.load_state_dict(torch.load('../saved_model_w/model0_bases.pth'))
init_embeddings.to(device)
print('Loaded RGCN layer to warm-start embeddings')
    

for batch_idx, (graph, pdbids, labels) in enumerate(train_loader):

    graph=send_graph_to_device(graph,device)
    labels = torch.tensor(labels).to(device)
    
    # Warm start embeddings 
    if(args.embeddings):
        init_embeddings.GNN(graph)
        
        # Updates g.ndata['h'] with the pretrained embeddings 
        
         
