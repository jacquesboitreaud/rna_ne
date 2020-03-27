import os, sys
import pickle

import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    sys.path.append("..")
    sys.path.append('data_processing')
    
from rna_classes import * 

from rna_layout import circular_layout




params= {'text.latex.preamble' : [r'\usepackage{fdsymbol}\usepackage{xspace}']}
plt.rc('font', family='serif')
plt.rcParams.update(params)

labels = {
    'CW': r"$\medblackcircle$\xspace",
    'CS': r"$\medblacktriangleright$\xspace",
    'CH': r"$\medblacksquare$\xspace",
    'TW': r"$\medcircle$\xspace",
    'TS': r"$\medtriangleright$\xspace",
    'TH': r"$\medsquare$\xspace"
}


make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2\
    else labels[s[:2]]

def rna_draw(nx_g, title="", highlight_edges=None, node_colors=None, num_clusters=None):
    # pos = circular_layout(nx_g)
    pos = nx.spring_layout(nx_g, k=0.1)

    nodes = nx.draw_networkx_nodes(nx_g, pos, node_size=150,  node_color='grey', linewidths=2)

    nodes.set_edgecolor('black')

    # plt.title(r"{0}".format(title))
    edge_labels = {}
    for n1,n2,d in nx_g.edges(data=True):
        try:
            symbol = make_label(d['label'])
            edge_labels[(n1, n2)] = symbol
        except:
            if d['label'] == 'B53':
                edge_labels[(n1, n2)] = ''
            else:
                edge_labels[(n1, n2)] = r"{0}".format(d['label'])
            continue

    non_bb_edges = [(n1,n2) for n1,n2,d in nx_g.edges(data=True) if d['label'] != 'B53']
    bb_edges = [(n1,n2) for n1,n2,d in nx_g.edges(data=True) if d['label'] == 'B53']

    nx.draw_networkx_edges(nx_g, pos, edgelist=non_bb_edges)
    nx.draw_networkx_edges(nx_g, pos, edgelist=bb_edges, width=2)

    if not highlight_edges is None:
        nx.draw_networkx_edges(nx_g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5)

    nx.draw_networkx_edge_labels(nx_g, pos, font_size=16,
                                 edge_labels=edge_labels)
    plt.axis('off')
    # plt.savefig('fmn_' + title + '.png', format='png')
    # plt.clf()
    plt.show()

def rna_draw_pair(graphs, title="", highlight_edges=None, node_colors=None, num_clusters=None):
    fig, ax = plt.subplots(1, len(graphs), num=1)
    for i,g in enumerate(graphs):
        pos = nx.spring_layout(g)

        if not node_colors is None:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150,  node_color=node_colors[i], linewidths=2, ax=ax[i])
        else:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150,  node_color='grey', linewidths=2, ax=ax[i])

        nodes.set_edgecolor('black')

        # plt.title(r"{0}".format(title))
        edge_labels = {}
        for n1,n2,d in g.edges(data=True):
            try:
                symbol = make_label(d['label'])
                edge_labels[(n1, n2)] = symbol
            except:
                if d['label'] == 'B53':
                    edge_labels[(n1, n2)] = ''
                else:
                    edge_labels[(n1, n2)] = r"{0}".format(d['label'])
                continue

        non_bb_edges = [(n1,n2) for n1,n2,d in g.edges(data=True) if d['label'] != 'B53']
        bb_edges = [(n1,n2) for n1,n2,d in g.edges(data=True) if d['label'] == 'B53']

        nx.draw_networkx_edges(g, pos, edgelist=non_bb_edges, ax=ax[i])
        nx.draw_networkx_edges(g, pos, edgelist=bb_edges, width=2, ax=ax[i])

        if not highlight_edges is None:
            nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[i])

        nx.draw_networkx_edge_labels(g, pos, font_size=16,
                                     edge_labels=edge_labels, ax=ax[i])
        ax[i].set_axis_off()

    plt.axis('off')
    plt.title(f"distance {title}")
    plt.show()
def generic_draw_pair(graphs, title="", highlight_edges=None, node_colors=None, num_clusters=None):
    fig, ax = plt.subplots(1, len(graphs), num=1)
    for i,g in enumerate(graphs):
        pos = nx.spring_layout(g)

        if not node_colors is None:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150,  node_color=node_colors[i], linewidths=2, ax=ax[i])
        else:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150,  node_color='grey', linewidths=2, ax=ax[i])

        nodes.set_edgecolor('black')

        # plt.title(r"{0}".format(title))
        edge_labels = {}
        for n1,n2,d in g.edges(data=True):
            edge_labels[(n1, n2)] = str(d['label'])

        if not highlight_edges is None:
            nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[i])

        nx.draw_networkx_edge_labels(g, pos, font_size=16,
                                     edge_labels=edge_labels, ax=ax[i])
        ax[i].set_axis_off()

    plt.axis('off')
    plt.title(f"distance {title}")
    plt.show()
    
    
    
def ablation_draw():
    g_name = "../data/chunks/1aq4.pickle"
    
    bads = []
    
    remove_stackings = True
    merge_stackings = False

    g = pickle.load(open(g_name, 'rb'))
    e=nx.get_edge_attributes(g,'label')
    for u,v,e in g.edges(data=True):
        print(u,v,e)
        if(e['label'] not in ('B35', 'B53')):
            if(e['label'] in ('S35','S53','S55','S33') and remove_stackings):
                bads.append((u,v))
            elif(merge_stackings):
                e['label']='CWW'
            elif(e['label'] not in ('S35','S53','S55','S33')):
                e['label']='CWW'
    g.remove_edges_from(bads)
    print(e)
    rna_draw(g, title='')

if __name__ == "__main__":
    ablation_draw()
