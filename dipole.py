from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import subprocess
from subprocess import DEVNULL
import os, sys
from pathlib import Path
import tempfile

import rdkit
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import numpy as np


@contextmanager
def suppress_output(verbose):
    """Suppress output when """
    if verbose:
        pass
    else:
        with open(devnull, 'w') as fnull:
            with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                yield (err, out)

def run_command(command, verbose):
    if verbose:
        subprocess.run(command, shell=True)
    else:
        subprocess.run(command, shell=True, stdout=DEVNULL, stderr=DEVNULL)


def get_dipole_moment(smile, verbose=False, scratch: str='/tmp'): 
    # Create and switch to temporary directory
    owd = Path.cwd()
    scratch_path = Path(scratch)
    tmp_dir = tempfile.TemporaryDirectory(dir=scratch_path)
    os.chdir(tmp_dir.name)

    # Create mol object
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    if mol == None: 
        return "INVALID"
    charge = Chem.rdmolops.GetFormalCharge(mol)
    atom_number = mol.GetNumAtoms()

    sas = sascorer.calculateScore(mol)
    
    with open('test.smi', 'w') as f: 
        f.writelines([smile])

    system = lambda x: run_command(x, verbose)
    
    # Prepare the input file: 
    system('obabel test.smi --gen3D -O test.xyz')

    # Run the preliminary xtb: 
    command_pre = 'CHARGE={};xtb {} --gfn 0 --opt normal -c $CHARGE --iterations 4000'.format(charge, 'test.xyz')
    system(command_pre)
    system("rm ./gfnff_charges ./gfnff_topo")

    # Run crest conformer ensemble
    command_crest = 'CHARGE={};crest {} -gff -mquick -chrg $CHARGE --noreftopo'.format(charge, 'xtbopt.xyz')
    system(command_crest)
    system('rm ./gfnff_charges ./gfnff_topo')
    system('head -n {} crest_conformers.xyz > crest_best.xyz'.format(atom_number+2))

    # Run the calculation: 
    command = 'CHARGE={};xtb {} --opt normal -c $CHARGE --iterations 4000 > out_dump'.format(charge, 'crest_best.xyz')
    system(command)

    # Read the output: 
    with open('./out_dump', 'r') as f: 
        text_content = f.readlines()

    output_index = [i for i in range(len(text_content)) if 'Property Printout' in text_content[i]]
    text_content = text_content[output_index[0]: ]
    homo_data = [x for x in text_content if '(HOMO)' in x]
    lumo_data = [x for x in text_content if '(LUMO)' in x]
    homo_lumo_gap = [x for x in text_content if 'HOMO-LUMO GAP' in x]
    mol_dipole    = [text_content[i:i+4] for i,x in enumerate(text_content) if 'molecular dipole:' in x]
    lumo_val      = float(lumo_data[0].split(' ')[-2])
    homo_val = float(homo_data[0].split(' ')[-2])
    homo_lumo_val  = float(homo_lumo_gap[0].split(' ')[-5])
    mol_dipole_val = float(mol_dipole[0][-1].split(' ')[-1])

    os.chdir(owd)
    tmp_dir.cleanup()

    return mol_dipole_val

if __name__ == "__main__":
    print(get_dipole_moment('CCO'))