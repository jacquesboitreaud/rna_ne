# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:09:30 2019

@author: jacqu

Utils functions for working with PDB files using biopython
"""

from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer

import numpy as np


def read_pdb(pdb_id):
    """ 
    describe inputs and outputs
    """
    parser = MMCIFParser()
    structure_id=pdb_id[:4]
    filename = f'../../data/rcsb_pdb/{structure_id}.cif/{structure_id}.cif'
    struc= parser.get_structure(structure_id, filename)
    residues = []
    for residue in struc.get_residues():
        if(residue.get_resname() in ['A','U','G','C']):
            residues.append(residue)
    return residues

def compute_rmsd(sup, coords_1, coords_2):
    """
    Computes RMSD after aligning atoms of struc1 to atoms of struc2
    """
    sup.set(coords_1, coords_2)
    sup.run()
    rms = sup.get_rms()
    return rms

def center(nt):
    """
    Computes coordinates of the center of heavy atoms of a nucleotide object
    Returns array of shape (3,)
    """
    c = np.zeros(3)
    N_c=len(nt.atoms)
    for a in nt.atoms:
        c[0]+=float(a.x)
        c[1]+=float(a.y)
        c[2]+=float(a.z)
    return c/N_c


    