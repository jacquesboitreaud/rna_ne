# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:08:55 2019

@author: jacqu
"""
from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.NeighborSearch import NeighborSearch

class PDB_structure(object):
    
    def __init__(self, pdb_id):
        parser = MMCIFParser()
        structure_id=pdb_id[:4]
        filename = f'../../data/rcsb_pdb/{structure_id}.cif/{structure_id}.cif'
        
        self.structure = parser.get_structure(structure_id, filename)
        r=[]
        for residue in self.structure.get_residues():
            if(residue.get_resname() in ['A','U','G','C']):
                r.append(residue)
        self.residues = r
        
        