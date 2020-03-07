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

import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

    from data_processing.pdb_utils import *
    from utils import *

def angle(ba,bc):
    """ radians angle between two vectors (numpy array, shape (3,))"""
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

def center(atoms):
    """ Computes center of mass given atoms of a base and returns coordinates as tuple (x,y,z)"""
    base_atoms = {'N1', 'C2', 'N3', 'C4', 'C5','C6','O2', 'O4', 'N4'} # pyrimidine atoms 
    base_atoms.update({'N2', 'O6', 'N6', 'N7','C8', 'N9'}) # purine specific ones 
    cx,cy,cz = 0,0,0
    cpt =0
    for a in atoms : 
        if(a.atom_label in base_atoms):
            cx += float(a.x)
            cy += float(a.y)
            cz += float(a.z)
            cpt+=1
    return (cx/cpt, cy/cpt, cz/cpt)

def base_angles(nucleotide, unit = 'rad'):
    """
    Takes a nucleotide object and returns defined angles values. unit = 'rad' or 'deg'.
    Out type : tuple 
    
    TODO:
    alpha:   O3'(i-1)-P-O5'-C5'
    beta:    P-O5'-C5'-C4'
    gamma:   O5'-C5'-C4'-C3'
    delta:   C5'-C4'-C3'-O3'
    epsilon: C4'-C3'-O3'-P(i+1)
    zeta:    C3'-O3'-P(i+1)-O5'(i+1)
    chi : glyc bond torsion
    base_gly : angle btw gly bond and base vector (NG, G is base geom center)
    
     """
    atoms=nucleotide.atoms
    for a in atoms : 
        if a.atom_label =="C1'":
            c1p = a 
        elif a.atom_label =="C3'":
            c3p = a 
        elif a.atom_label =="O3'":
            o3p = a 
        elif a.atom_label == "O4'":
            o4p=a
        elif a.atom_label =="C4'":
            c4p = a 
        elif a.atom_label =="C5'":
            c5p = a
        elif a.atom_label =="O5'":
            o5p = a  
        elif a.atom_label =="P":
            P = a
    
    if (nucleotide.real_nt in ('G','A')): # Purine, chi = O4'-C1' // N9-C4
        
        n = [a for a in atoms if a.atom_label =="N9"][0]
        c = [a for a in atoms if a.atom_label=='C4'][0]
    
    elif (nucleotide.real_nt in ('U','C')): #Pyrimidine , chi = O4'-C1' // N1-C2
        
        n = [a for a in atoms if a.atom_label =="N1"][0]
        c = [a for a in atoms if a.atom_label=='C2'][0]
        
    else:
        print(f'!!!! Nucleotide type not handled: {nucleotide}')
        
    coords = np.zeros((2,3))
    
    # Chi torsion angle 
    # vector O4'-c1'
    coords[0]=float(c1p.x)-float(o4p.x), float(c1p.y)-float(o4p.y), float(c1p.z)-float(o4p.z) 
    # vector N-C 
    coords[1]=float(c.x)-float(n.x), float(c.y)-float(n.y), float(c.z)-float(n.z)
    chi = angle(coords[0], coords[1])
    
    # Delta torsion angle 
    # vector C5'-C4'
    coords[0]=float(c4p.x)-float(c5p.x), float(c4p.y)-float(c5p.y), float(c4p.z)-float(c5p.z)
    # vector C3'-O3'
    coords[1]=float(o3p.x)-float(c3p.x), float(o3p.y)-float(c3p.y), float(o3p.z)-float(c3p.z)
    delta = angle(coords[0], coords[1])
    
    # Angle btw glycosidic bond (NC1') and base (NG, G center of base)
    # vector NC1'
    coords[0] = float(c1p.x)-float(n.x), float(c1p.y)-float(n.y), float(c1p.z)-float(n.z) 
    #vector NG
    gx, gy, gz = center(atoms)
    coords[1]=float(gx)-float(n.x), float(gy)-float(n.y), float(gz)-float(n.z)
    base_gly = angle(coords[0], coords[1])
    
    
    if(unit == 'deg'): # convert to degrees 
        chi = chi *180/np.pi
        delta = delta*180/np.pi
        base_gly = base_gly*180/np.pi
        
    return chi, delta, base_gly


if __name__=='__main__':
    
    # Load a sample graph 
    gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
    graphs = os.listdir(gr_dir)
    pid=graphs[0]
    g=pickle.load(open(os.path.join(gr_dir,pid), 'rb'))
    
    for node, data in g.nodes(data=True):
        
            nucleotide = data['nucleotide']
            print('nt angles: ')
            print(base_angles(nucleotide, 'deg'))
            print('geometric center of base: ')
            print(center(nucleotide.atoms))
        

       
            


