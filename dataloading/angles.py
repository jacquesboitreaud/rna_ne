 # -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:52:10 2019

@author: jacqu

Computing angles for backbone and base, for 1 nt 
"""

import numpy as np
import itertools
import pickle 
import os 
import networkx as nx
from rna_classes import *
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBIO
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from dataloading.pdb_utils import *
from utils import *
from rotation import rot 

def angle(ba,bc):
    """ radians angle between two vectors (numpy array, shape (3,))"""
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

def chi(nucleotide, unit = 'rad'):
    # Takes a nucleotide object and returns chi angle (glycosidic-backbone) value (scalar)
    atoms=nucleotide.atoms
    
    if (nucleotide.nt in ('G','A')): # Purine, O4'-C1' // N9-C4

        o4_p = [a for a in atoms if a.atom_label =="O4'"][0]
        c1_p = [a for a in atoms if a.atom_label =="C1'"][0]
        
        n = [a for a in atoms if a.atom_label =="N9"][0]
        c = [a for a in atoms if a.atom_label=='C4'][0]
    
    if (nucleotide.nt in ('U','C')): #Pyrimidine , O4'-C1' // N1-C2
        o4_p = [a for a in atoms if a.atom_label =="O4'"][0]
        c1_p = [a for a in atoms if a.atom_label =="C1'"][0]
        
        n = [a for a in atoms if a.atom_label =="N1"][0]
        c = [a for a in atoms if a.atom_label=='C2'][0]
        
    coords = np.zeros((2,3))
    
    # vector O4'-c1'
    coords[0]=float(c1_p.x)-float(o4_p.x), float(c1_p.y)-float(o4_p.y), float(c1_p.z)-float(o4_p.z) 
    
    # vector N-C 
    coords[1]=float(c.x)-float(n.x), float(c.y)-float(n.y), float(c.z)-float(n.z)
    
    # Chi torsion angle 
    chi = angle(coords[0], coords[1])
    
    if(unit == 'deg'): # convert to degrees 
        chi = chi *180/np.pi
        
    return chi

#TODO ; compute backbone torsions and/or sugar ring torsions .? 


if __name__=='__main__':
    
    # Load a sample graph 
    gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
    graphs = os.listdir(gr_dir)
    pid=graphs[0]
    g=pickle.load(open(os.path.join(gr_dir,pid), 'rb'))
    
    print('printing chi torsion angles, in degrees : ')
    for node, data in g.nodes(data=True):
        
            nucleotide = data['nucleotide']
            print(chi(nucleotide, 'deg'))
        

       
            


