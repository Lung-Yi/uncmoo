from tartarus.docking import apply_filters

from rdkit import Chem
import rdkit.Chem.rdmolops as rdcmo
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import os

import rdkit
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

"""
The penalty file indicating what types of molecules should be avoided during molecular design.
If molecules containing the unfavorable substurctures, then the filter function would return True.
"""
def calc_aromaticity_degree(mol):
    return len(list(mol.GetAromaticAtoms())) / mol.GetNumAtoms()

def calc_conjugation_degree(mol):
    return sum([int(bond.GetIsConjugated()) for bond in mol.GetBonds()]) / mol.GetNumBonds()

def maximum_minimum_ring_size(mol):
    """
    Calculate maximum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        maximum_ring_size = 0
        minimum_ring_size = 0
    else:
        maximum_ring_size = max([len(ci) for ci in cycles])
        minimum_ring_size = min([len(ci) for ci in cycles])
    return maximum_ring_size, minimum_ring_size

def organic_emitter_filter(smiles):
    def substructure_violations_organic_emitter(mol):
        """
        Check for substructure violates
        Return True: contains a substructure violation
        Return False: No substructure violation
        """
        violation = False
        forbidden_fragments = ["[Cl,Br,I]", "*=*=*", "*#*", "[O,o,S,s]~[O,o,S,s]", 
                               "[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]", "[C,c]~N=,:[O,o,S,s;!R]",
                               "[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]",
                               "*=[NH]", "*=N-[*;!R]", "*~[N,n,O,o,S,s]-[N,n,O,o,S,s;!R]",
                               "*-[CH1]-*", "*-[CH2]-*", "*-[CH3]"]
        for ni in range(len(forbidden_fragments)):
            if mol.HasSubstructMatch(Chem.MolFromSmarts(forbidden_fragments[ni])) == True:
                violation = True
                break
            else:
                continue
        return violation
    
    mol = Chem.MolFromSmiles(smiles)
    maximum_ring_size, minimum_ring_size = maximum_minimum_ring_size(mol)
    if mol.GetNumAtoms() > 23:
        return True
    elif rdcmo.GetFormalCharge(mol) != 0:
        return True
    elif rdcd.NumRadicalElectrons(mol) != 0:
        return True
    elif rdcmd.CalcNumBridgeheadAtoms(mol) > 0:
        return True
    elif rdcmd.CalcNumSpiroAtoms(mol) > 0:
        return True
    elif calc_aromaticity_degree(mol) < 0.5:
        return True
    elif calc_conjugation_degree(mol) < 0.7:
        return True
    elif maximum_ring_size not in [5, 6, 7, 8]:
        return True
    elif minimum_ring_size not in [5, 6, 7, 8]:
        return True
    elif substructure_violations_organic_emitter(mol): # cannot contain some functional groups
        return True
    elif sascorer.calculateScore(mol) > 4.5: # need to generate molecules under SAscore constraint 
        return True

    return False


def docking_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumAtoms() > 48:
        return True
    return not apply_filters(smiles)

def dockstring_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumAtoms() > 60:
        return True
    return not apply_filters(smiles)

def reactivity_filter(smiles):
    def substructure_preserver(mol):
        """
        Check for substructure violates
        Return True: contains a substructure violation
        Return False: No substructure violation
        """        
        mol = rdkit.Chem.rdmolops.AddHs(mol) # Note: Hydrogens need to be added for the substructure code to work!
        
        if mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts('[H][C@@]1(*)[C@;R2](*)2[C@@]34[C@;R2]5(*)[C;R1](*)=[C;R1](*)[C@;R2](*)([*;R2]5)[C@@]3([*;R1]4)[C@](*)([*;R2]2)[C@@;R1]1([*])[H]')) == True:
            return True # Has substructure! 
        else: 
            return False # Molecule does not have substructure!
    
    def substructure_violations(mol):
        """
        Check for substructure violates
        Return True: contains a substructure violation
        Return False: No substructure violation
        """
        violation = False
        forbidden_fragments = ['[C-]', '[S-]', '[O-]', '[N-]', '[*+]', '[*-]', '[PH]', '[pH]', '[N&X5]', '*=[S,s;!R]', '[S&X3]', '[S&X4]', '[S&X5]', '[S&X6]', '[P,p]', '[B,b,N,n,O,o,S,s]~[F,Cl,Br,I]', '*=*=*', '*#*', '[O,o,S,s]~[O,o,S,s]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]', '*=N-[*;!R]', '*~[N,n,O,o,S,s]-[N,n,O,o,S,s;!R]']
        for ni in range(len(forbidden_fragments)):
            
            if mol.HasSubstructMatch(Chem.MolFromSmarts(forbidden_fragments[ni])) == True:
                violation = True
                break
            else:
                continue
        return violation

    mol = Chem.MolFromSmiles(smiles)
    if mol == None:
        return True
    elif not substructure_preserver(mol): # need to preserve the core substructure for the specific reaction
        return True
    elif substructure_violations(mol): # cannot contain some functional groups
        return True
    elif sascorer.calculateScore(mol) > 6.0: # need to generate molecules under SAscore constraint 
        return True
    elif mol.GetNumAtoms() > 55:
        return True
    return False

if __name__ == "__main__":
    smiles = "CCCCCCCCC"
    print(organic_emitter_filter(smiles))
    # smiles = "C=C(CCl)CC1OC(C=COC2C(O)C3CC2C24OC32C2C=CC4N2)C2CC(C)C13OC23CC"
    smiles = "CCCC1(O)CC2C=C3C(OC(C)CCCCCOC4C5=C(OCC=COC6CC7CC6C68OC76C6C=CC8N6)C54)NN2C31"
    mol = Chem.MolFromSmiles(smiles)
    print(sascorer.calculateScore(mol))
    print(reactivity_filter(smiles))