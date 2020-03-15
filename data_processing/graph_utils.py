# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:07:45 2020

@author: jacqu

Utils functions for RNA graph processing and context selection 
"""

import networkx as nx
import itertools
    
    
    
def nodes_within_radius(G, u_idx, inner, outer) :
    """ 
    Takes graph, node index (in sorted(g.nodes)) and inner and outer ring radii
    Build the context graph around node [u_idx] and returns graph object 
    """
    depth = outer+1 
    nodes = sorted(G.nodes())
    total_nodes = [list([nodes[u_idx]])] # list of lists, nodes at dist k of the source node 
    assert(len(total_nodes)>0)

    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
        total_nodes.append(depth_ring)
            
    if(inner>0):
        total_nodes = total_nodes[inner:] # Remove rings closer to source than the inner radius

    return set(itertools.chain(*total_nodes))


def find_node(graph, chain, pos):
    # Find a node in a networkx rna graph 
    for n,d in graph.nodes(data=True):
        if (n[0] == chain) and (d['nucleotide'].pdb_pos == str(pos)):
            return n
    return None

def dangle_trim(G):

    """
    Recursively remove dangling nodes from graph.
    """

    cur_G = G.copy()

    while True:
        dangles = []
        for n in cur_G.nodes:
            # node_deg = degree(i, G, current_nodeset)
            # print(node_deg)
            # if node_deg == 2 and is_backbone(n, G):
            # if cur_G.degree(n) == 1 and is_backbone(n, cur_G) or cur_G.degree(n) == 0:
            if cur_G.degree(n) == 1  or cur_G.degree(n) == 0:
                dangles.append(n)
        if len(dangles) == 0:
            break
        else:
            cur_G.remove_nodes_from(dangles)
            cur_G = cur_G.copy()
    return cur_G

