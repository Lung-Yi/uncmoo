import torch
from .chemprop_base_model import ChempropUncertaintyPredictor
from .penalty import organic_emitter_filter, docking_filter, reactivity_filter, dockstring_filter

    
class DockingScorePredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if docking_filter(smiles):
            return -100000
        else:
            return 0
        
class OrganicEmitterScorePredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="mve", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if organic_emitter_filter(smiles):
            return -100000
        else:
            return 0

class ReactivityPredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if reactivity_filter(smiles):
            return -100000
        else:
            return 0
        
class DockstringPredictor(ChempropUncertaintyPredictor):
    def __init__(self,model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_path, uncertainty_method="evidential_total", batch_size=batch_size, device=device)

    def penalty(self, smiles):
        if dockstring_filter(smiles):
            return -100000
        else:
            return 0