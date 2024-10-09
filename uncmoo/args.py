from typing import List, Optional
from typing_extensions import Literal
from tap import Tap

class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`Janus` and :class:`PredictArgs`."""
    benchmark_dataset: Literal['docking', 'organic_emitter', 'hce_advanced', 'hce_simple', 'reactivity', 'dockstring', 'similarity_Aripiprazole',
                               'similarity_Albuterol', 'similarity_Mestranol', 'median_molecule_1', 'median_molecule_2', 'mpo_Fexofenadine', 
                               'mpo_Ranolazine']
    """Benchmark dataset."""
    fitness_method: Literal['uncertainty', 'scalarization', 'scaler', 'utopian', 'hybrid', 'expected_improvement']
    """What method to calculate the fitness."""
    n_sample: int = 1000
    """Number of starting molecules used in the pool of genetic algoerithm."""
    result_path: str = None
    """Path to where the optimization results would be saved."""
    sample_data_path: str = None
    """Path to where the SMILES file used for initializing the genetic pool."""
    start_smiles_path: str = None
    """Path to where the starting SMILES is saved for the generative model."""
    surrogate_model_path: List[str] = None
    """Path to the trained surrogate model. (chemprop uncertainty model is .pt file)"""
    target_columns: List[str] = None
    """The name of targets. They should match with the names saved in the chemprop models."""
    calibration_factors: List[float] = None
    """The calibration factors used to divide the uncertainty prediction values. """
    target_cutoff: List[float] = None
    """The cutoffs of the target for calculating CDF."""
    target_objective: List[str] = None
    """The criteria of the targets. Only accepts minimization / maximization."""
    target_weight: List[float] = None
    """The weight of each target for the scalarization method. The input order should match with the target_columns."""
    target_scaler: List[float] = None
    """The scaler of each target for the scaler method.
    The input order should be target1_mean, target1_std, target2_mean, target2_std..."""
    target_utopian: List[float] = None
    """The scaler of each target for the utopian method.
    The input order should be target1_utopian, target1_std, target2_utopian, target2_std..."""
    batch_pred: bool = False
    """Batch prediction mode"""
    alphabet_path: str = None

class JanusArgs(CommonArgs):
    """:class:`JanusArgs` includes :class:`CommonArgs` along with additional arguments used for Janus genetic algorithm."""
    generations: int = 10
    """Number of iterations that JANUS runs for."""
    generation_size: int = 500
    """The number of molecules for which fitness calculations are done, exploration and exploitation each have their own population"""
    num_exchanges: int = 5
    """Number of molecules that are exchanged between the exploration and exploitation."""
    num_workers: int = 8
    """For multi-processing."""
    use_fragments: bool = True
    """Fragments from starting population used to extend alphabet for mutations."""
    use_classifier: bool = True
    """An option to use a classifier as selection bias."""
    top_mols: int = 1
    """Number of the best molecules printed in each generation."""
    verbose_out: bool = False
    """Whether to print detailed information."""
    # alphabet: Optional[List[str]] = None,
    explr_num_random_samples: Optional[int] = 5
    explr_num_mutations: Optional[int] = 5
    crossover_num_random_samples: Optional[int] = 1
    exploit_num_random_samples: Optional[int] = 400
    exploit_num_mutations: Optional[int] = 400
    use_gpu: Optional[bool] = True,
    # early_stop_patience: int = 30,
    """Default setting."""
    def __init__(self, *args, **kwargs) -> None:
        super(CommonArgs, self).__init__(*args, **kwargs)
