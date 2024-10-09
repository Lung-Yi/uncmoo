import torch
from rdkit import Chem
from .chemprop_base_model import ChempropUncertaintyPredictor, MultipleChempropUncertaintyPredictor
from .penalty import organic_emitter_filter, docking_filter, reactivity_filter, dockstring_filter

    
class DockingScorePredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, calibration_factors=None, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total", calibration_factors=None, batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if docking_filter(smiles):
            return -100000
        else:
            return 0
        
class OrganicEmitterScorePredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, calibration_factors=None, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="mve", calibration_factors=None, batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if organic_emitter_filter(smiles):
            return -100000
        else:
            return 0

class ReactivityPredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, uncertainty_method="evidential_total", calibration_factors=None, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method=uncertainty_method,calibration_factors=calibration_factors, batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if reactivity_filter(smiles):
            return -100000
        else:
            return 0
        
class DockstringPredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, calibration_factors=None, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total",calibration_factors=None, batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if dockstring_filter(smiles):
            return -100000
        else:
            return 0

class SimilarityPredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, calibration_factors=None, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total",calibration_factors=None, batch_size=batch_size, device=device)

    def penalty(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() > 40:
            return -100000
        return 0

class MultiBenchmarkPredictor(MultipleChempropUncertaintyPredictor):
    """Used for the multi-objective benchmark in Guacamol. Each objective is predicted by a single-task Chemprop model. """
    def __init__(self,model_paths, uncertainty_methods, calibration_factors=None, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_paths, uncertainty_methods=uncertainty_methods,calibration_factors=None, batch_size=batch_size, device=device)

    def penalty(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() > 40:
            return -100000
        return 0