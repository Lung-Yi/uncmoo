import numpy as np
import torch
from scipy.stats import norm
from rdkit import Chem
import rdkit.Chem.rdmolops as rdcmo
import rdkit.Chem.Descriptors as rdcd
import os

import rdkit
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from chemprop.data import get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.train import predict
from Tartarus.tartarus.docking import apply_filters
from .chemprop_base_model import ChempropEnsembleMVEPredictor, ChempropEvidentialUncertaintyPredictor, ChempropUncertaintyPredictor
from .penalty import organic_emitter_filter, docking_filter, reactivity_filter

    
class DockingScorePredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if docking_filter(smiles):
            return -10000
        else:
            return 0
        
class OrganicEmitterScorePredictor(ChempropUncertaintyPredictor): # ChempropEvidentialUncertaintyPredictor
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="mve", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if organic_emitter_filter(smiles):
            return -10000
        else:
            return 0
        
class SeparateOrganicEmitterScorePredictor():
    def __init__(self, model_path_dict, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.batch_size = batch_size
        self.device = device
        self.unc_model_dict = {}
        for task_name, model_path in model_path_dict.items():
            unc_model = ChempropEvidentialUncertaintyPredictor(model_path, batch_size, device)
            if type(task_name) == str:
                assert task_name == unc_model.task_names[0]
            else: raise ValueError("Not implement separate multi-task model yet.")
            self.unc_model_dict.update({task_name: unc_model})

    def calc_overall_fitness(self, smiles_list):
        overall_fitness = 1.
        for task_name, unc_model in self.unc_model_dict.items():
            multiobjective_fitness = unc_model.calc_multiobjective_fitness(smiles_list)
            overall_fitness *= np.prod(multiobjective_fitness, axis=0)
        return overall_fitness
    
    def load_target_cutoff(self, target_cutoff_dict, target_objective_dict):
        for task_name, unc_model in self.unc_model_dict.items():
            unc_model.load_target_cutoff(target_cutoff_dict, target_objective_dict)
        return
    
    def load_target_weights(self, target_weight_dict):
        for task_name, unc_model in self.unc_model_dict.items():
            unc_model.load_target_weights(target_weight_dict)
        return
    
    def load_target_scaler(self, target_scaler_dict, target_objective_dict):
        for task_name, unc_model in self.unc_model_dict.items():
            unc_model.load_target_scaler(target_scaler_dict, target_objective_dict)
        return
    
    def batch_oe_penalty(self, smiles_list):
        return np.array([self.oe_penalty(smiles) for smiles in smiles_list])
    
    def oe_penalty(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # if "P" in smiles.upper():
        #     return -10000
        if mol.GetNumAtoms() > 23:
            return -10000
        # elif IsRadicalOrCharge(mol):
        #     return -10000
        else:
            return 0
        
    def batch_uncertainty_fitness(self, smiles_list):
        fitness = 1.
        for task_name, unc_model in self.unc_model_dict.items():
            fitness *= unc_model.calc_overall_fitness(smiles_list)
        return fitness + self.batch_oe_penalty(smiles_list)
    
    def uncertainty_fitness(self, smiles):
        fitness = 1.
        for task_name, unc_model in self.unc_model_dict.items():
            fitness *= unc_model.single_overall_fitness(smiles)
        return fitness + self.oe_penalty(smiles)
    
    def batch_scalarization_fitness(self, smiles_list):
        fitness = 0.
        for task_name, unc_model in self.unc_model_dict.items():
            fitness += unc_model.calc_scalarization_fitness(smiles_list)
        return fitness + self.batch_oe_penalty(smiles_list)

    def scalarization_fitness(self, smiles):
        fitness = 0
        for task_name, unc_model in self.unc_model_dict.items():
            fitness += unc_model.single_scalarization_fitness(smiles)
        return fitness + self.oe_penalty(smiles)
    
    def batch_scaler_fitness(self, smiles_list):
        fitness = 0.
        for task_name, unc_model in self.unc_model_dict.items():
            fitness += unc_model.calc_scaler_fitness(smiles_list)
        return fitness + self.batch_oe_penalty(smiles_list)

    def scaler_fitness(self, smiles):
        fitness = 0
        for task_name, unc_model in self.unc_model_dict.items():
            fitness += unc_model.single_scaler_fitness(smiles)
        return fitness + self.oe_penalty(smiles)
     
class ReactivityPredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if reactivity_filter(smiles):
            return -10000
        else:
            return 0