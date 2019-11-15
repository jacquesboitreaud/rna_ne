# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:00:41 2019

@author: jacqu

Functions to read pdb with biopython and keep atoms around a nucleotide 
"""


from Bio import PDB
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

parser = parser = MMCIFParser()


# Load pdb structure
structure_id = "6n2v"
filename = f'../../data/rcsb_pdb/{structure_id}.cif/{structure_id}.cif'
mmcif = MMCIF2Dict(filename)

"""
# Load some info from mmcif dict: 
print(mmcif.keys())
print('Nb nucleic acid atoms: ',mmcif[ '_refine_hist.pdbx_number_atoms_nucleic_acid'])
"""

print(mmcif['_entity.pdbx_fragment'])






"""
# Iterate on residues 
for residue in structure.get_residues():
    print(residue)
"""