# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:09:30 2019

@author: jacqu

Utils functions for working with PDB files using biopython
"""

from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SVDSuperimposer import SVDSuperimposer

from Bio.PDB.PDBIO import Select

import numpy as np


def read_pdb(pdb_id, path):
    """ 
    describe inputs and outputs
    """
    parser = MMCIFParser()
    struc= parser.get_structure(pdb_id, path)
    """
    residues = []
    for residue in struc.get_residues():
        if(residue.get_resname() in ['A','U','G','C']):
            residues.append(residue)
    """
    return struc

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

class selectResidues(Select):
    """ Selects residues by their pdb_position attribute to write into PDB (PDBIO writer object)"""
    def __init__(self, positions):
        super(selectResidues, self).__init__()
        self.positions = positions
        
    def accept_residue(self, residue):
        # Accepts residues in pdb writer 
        #print(residue.get_id()[1])
        if residue.get_id()[1] in self.positions:
            return 1
        else:
            return 0
        
def get_score(outfile):
    # Give path to output file
    with open(outfile,'r') as f:
        lines = f.readlines()
        
        #First catch: if file is empty (one sequence too short):
        if(len(lines)==0):
            print("Empty file")
            return -1
        else:
            if('TM-score' not in lines[13]):
                print('!!! Bad Line index (13) !!')
                return -1
            if('TM-score' not in lines[14]):
                print('!!! Bad Line index (14) !!')
                return -1
            tm1, tm2 = float(lines[13].split()[1]), float(lines[14].split()[1])
            return (tm1+tm2)/2


    