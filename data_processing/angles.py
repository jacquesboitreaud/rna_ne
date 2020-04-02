 # -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:52:10 2019

@author: jacqu

Computing angles for backbone and base, for 1 nt 

New version : all angles + proper computation 
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

    from utils import *
    
def norm(vec):
    # Norms a vector
    return vec / np.linalg.norm(vec)

def ortho(vec,ref):
    # Returns orthogonal component of vec to ref 
    # ! Ref should already be a normed vector
    """
    if(np.abs(np.linalg.norm(ref)-1)>0.001): # Ref is not a normed vector : norm it 
        ref = norm(ref)
    """
    vprime = vec - ref * np.dot(vec,ref)
    return vprime/np.linalg.norm(vprime)
    
def angle(ba,bc):
    """ radians angle between two unnormed vectors (numpy array, shape (3,))"""
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

def torsion(a,b,c,d, unit = 'rad'):
    # Torsion angle between bonds ab and cd , around axis bc. 
    # Input are atoms from rna classes format (a.x, a.y, a.z coordinates)
    
    # norm bc : 
    bc = float(c.x)-float(b.x), float(c.y)-float(b.y), float(c.z)-float(b.z)
    bc = norm(bc)
    
    # vectors ba and cd : 
    ba = float(a.x)-float(b.x), float(a.y)-float(b.y), float(a.z)-float(b.z)
    cd = float(d.x)-float(c.x), float(d.y)-float(c.y), float(d.z)-float(c.z)
    
    # Orthogonal components of ba and cd wrt. axis bc : 
    ba_O = ortho(ba, bc)
    cd_O = ortho(cd, bc)
    
    # Compute angle and sign
    angle_rad = np.arccos(np.dot(ba_O, cd_O)) 
    sign = np.dot(np.cross(ba_O, cd_O), bc)  
    if (sign < 0):
        angle_rad = -angle_rad
    
    # Output in desired unit 
    if(unit=='deg'):
        return angle_rad *180 /np.pi
    else:
        return angle_rad

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

def base_angles(nucleotide, nt_prev=None, nt_next=None):
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
    
    # Set all angles to zero by default (will be used as error marker)
    alpha, beta, gamma, delta, epsilon, zeta, chi, base_gly = [0.0 for i in range(8)]
     
    # 1/ Angles with only current nucleotide 
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
    
    if (nucleotide.nt in ('G','A')): # Purine, chi = O4'-C1' // N9-C4
        
        n_base = [a for a in atoms if a.atom_label =="N9"]
        c_base = [a for a in atoms if a.atom_label=='C4']
        
        if len(n_base)==1:
            n_base = n_base[0]
        if len(c_base)==1:
            c_base = c_base[0]
    
    elif (nucleotide.nt in ('U','C')): #Pyrimidine , chi = O4'-C1' // N1-C2
        
        n_base = [a for a in atoms if a.atom_label =="N1"]
        c_base = [a for a in atoms if a.atom_label=='C2']
        
        if len(n_base)==1:
            n_base = n_base[0]
        if len(c_base)==1:
            c_base = c_base[0]
    
    #beta:    P-O5'-C5'-C4'
    try:
        beta = torsion (P, o5p, c5p, c4p)
    except:
        pass
    
    #gamma:   O5'-C5'-C4'-C3'
    try:
        gamma = torsion(o5p, c5p, c4p, c3p)
    except:
        pass
    
    # Delta : C5'-C4'-C3'-O3'
    try:
        delta = torsion(c5p, c4p, c3p, o3p)
    except:
        pass
    
    # Chi torsion angle : O4p C1p Nbase Cbase
    try:
        chi = torsion(o4p, c1p, n_base, c_base)
    except:
        pass
    
    # Angle btw glycosidic bond (NC1') and base (NG, G center of base)
    try:
        #vector NC1'
        u = [float(c1p.x)-float(n_base.x), float(c1p.y)-float(n_base.y), float(c1p.z)-float(n_base.z) ]
        #vector NG
        gx, gy, gz = center(atoms)
        v=[float(gx)-float(n_base.x), float(gy)-float(n_base.y), float(gz)-float(n_base.z)]
        base_gly = angle(u, v)
    except:
        pass
    
    # 2/ Prev nucleotide
    alpha = 0 # error (default)
    if(nt_prev !=None):
        atoms = nt_prev.atoms
        o3p_prev = [a for a in atoms if a.atom_label =="O3'"]
        
        if(len(o3p_prev)==1):
            o3p_prev=o3p_prev[0]
            # alpha: O3'(i-1)-P-O5'-C5'
            try:
                alpha = torsion(o3p_prev, P, o5p, c5p)
            except:
                pass
            
    #3 / next nucleotide 
    epsilon, zeta = 0,0
    if(nt_next !=None):
        atoms = nt_next.atoms
        o5p_next = [a for a in atoms if a.atom_label =="O5'"]
        P_next = [a for a in atoms if a.atom_label =="P"]
        
        if len(P_next)==1:
            P_next = P_next[0]
            #epsilon: C4'-C3'-O3'-P(i+1)
            try:
                epsilon = torsion(c4p, c3p, o3p, P_next)
            except:
                pass
        
        if len(o5p_next)==1:
            o5p_next = o5p_next[0]
            #zeta: C3'-O3'-P(i+1)-O5'(i+1)
            try:
                zeta = torsion(c3p, o3p, P_next, o5p_next)
            except:
                pass
            
    return [alpha, beta, gamma, delta, epsilon, zeta, chi, base_gly] # 8 angles output 

def norm_base_angles(nucleotide):
    # Computes angles phi, psi of the normal vector to the base plane 
    atoms = nucleotide.atoms
    # center G 
    gx, gy, gz = center(atoms)
    
    if (nucleotide.nt in ('G','A')): # Purine, chi = O4'-C1' // N9-C4
        
        n_base = [a for a in atoms if a.atom_label =="N9"]
        
        if len(n_base)==1:
            n_base = n_base[0]
    
    elif (nucleotide.nt in ('U','C')): #Pyrimidine , chi = O4'-C1' // N1-C2
        
        n_base = [a for a in atoms if a.atom_label =="N1"]
        
        if len(n_base)==1:
            n_base = n_base[0]
            
    c5 = [a for a in atoms if a.atom_label =="C5"]
    if(len(c5)>0):
        c5 = c5[0]
    
    try:
        # vector G-N
        u = -1*np.array([float(gx)-float(n_base.x), float(gy)-float(n_base.y), float(gz)-float(n_base.z) ])
        # vector G-C5 
        v = -1*np.array([float(c5.x)-float(n_base.x), float(c5.y)-float(n_base.y), float(c5.z)-float(n_base.z) ])
        
        # normal vec 
        n = np.cross(u,v)
        n=norm(n)
    except:
        return 0,0
    
    r=np.sqrt(n[0]**2+n[1]**2+n[2]**2)
    # radial coordinates angles 
    phi = np.arctan(n[1]/n[0])
    theta = np.arccos(n[2]/r)
    
    return theta, phi
    
    
if __name__=='__main__':
    
    # Load a sample graph 
    gr_dir = "C:/Users/jacqu/Documents/MegaSync Downloads/RNA_graphs"
    graphs = os.listdir(gr_dir)
    pid=graphs[0]
    g=pickle.load(open(os.path.join(gr_dir,pid), 'rb'))
    
    for node, data in g.nodes(data=True):
        
            nucleotide = data['nucleotide']
            print('node id ', node[1])
            print('pdb position : ', nucleotide.pdb_pos)
            
            print('nt angles: ')
            print(norm_base_angles(nucleotide))

        

       
            


