import numpy as np
import torch
import tempfile
from abc import ABC, abstractmethod
from typing import List

from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.train import load_model, make_predictions
from chemprop.train import predict
from scipy.stats import norm

def gaussian_cdf(mean, variance, cutoff):
    std_dev = variance**0.5 
    dist = norm(mean, std_dev)  # construct the distribution
    cdf = dist.cdf(cutoff)  # probability that the distribution will be less than or equal to the cutoff
    return cdf

def expected_improvement(predictions, variances, cutoff, minimize=False):
    """
    Calculate the Expected Improvement (EI) for Bayesian Optimization, supporting both
    maximization and minimization objectives.

    :param predictions: 2D numpy array of mean predictions from the Gaussian process
    :param variances: 2D numpy array of variances from the Gaussian process
    :param cutoff: Scalar, the current best observed value (or a cutoff)
    :param minimize: Boolean, True if the objective is to minimize the function, False for maximization
    :return: 2D numpy array of Expected Improvement values
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if minimize:
            # For minimization, improvements are calculated as current best minus predictions
            improvements = cutoff - predictions
        else:
            # For maximization, improvements are predictions minus current best
            improvements = predictions - cutoff
        # Standard deviations
        std_devs = np.sqrt(variances)
        # Compute the Z value for the normal distribution
        Z = improvements / std_devs
        Z = np.where(std_devs > 0, Z, 0)  # Avoid division by zero
        # Calculate the EI
        ei = improvements * norm.cdf(Z) + std_devs * norm.pdf(Z)
        ei = np.where(std_devs > 0, ei, 0)  # EI is zero where std_dev is zero
    return ei

class ChempropUncertaintyPredictor():
    def __init__(self, model_path, uncertainty_method, calibration_factors:List[float]=None,
                 batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        arguments = [
            '--test_path', None,
            '--preds_path', tempfile.NamedTemporaryFile().name,
            '--checkpoint_dir', model_path,
            '--uncertainty_method', uncertainty_method,
        ]
        args = PredictArgs().parse_args(arguments)

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.args, self.train_args, self.models, self.scalers, self.num_tasks, self.task_names = load_model(args)
        for model in self.models:
            model = model.to(self.device)
        self.args.num_workers = 0
        if calibration_factors == None:
            self.calibration_factors = np.array([1.])
        else:
            self.calibration_factors = np.array(calibration_factors)
        print("calibration:", self.calibration_factors)

    def predict(self, smiles_list):
        mean, total_uncertainty = make_predictions(
                self.args,
                smiles=[[s] for s in smiles_list],
                model_objects=(self.args, self.train_args, self.models, self.scalers, self.num_tasks, self.task_names),
                calibrator = None,
                return_invalid_smiles = False,
                return_index_dict = False,
                return_uncertainty = True,
            )
        return np.array(mean), np.array(total_uncertainty) / self.calibration_factors
    
    def load_target_cutoff(self, target_cutoff_dict, target_objective_dict):
        assert set(target_objective_dict.values()).issubset(set(["maximize", "minimize"])) 
        self.target_cutoff_dict = target_cutoff_dict
        self.target_objective_dict = target_objective_dict
        return
    
    def load_target_weights(self, target_weight_dict):
        self.target_weight_dict = target_weight_dict
        return
    
    def load_target_scaler(self, target_scaler_dict, target_objective_dict):
        assert set(target_objective_dict.values()).issubset(set(["maximize", "minimize"]))
        self.target_scaler_dict = target_scaler_dict
        self.target_objective_dict = target_objective_dict
        return
    
    def load_utopian_objective(self, target_utopian_dict, target_objective_dict):
        assert set(target_objective_dict.values()).issubset(set(["maximize", "minimize"]))
        self.target_utopian_dict = target_utopian_dict
        self.target_objective_dict = target_objective_dict
        return

    def calc_expected_improvement_fitness(self, smiles_list):
        """ Calculate the expected improvement fitness. It only supports single-objective calculation. """
        if (len(self.target_objective_dict) > 1) or (len(self.target_cutoff_dict) > 1):
            raise ValueError("The expected improvement fitness only supports single-objective")
        preds, variance = self.predict(smiles_list)
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict.get(target)
            cutoff = self.target_cutoff_dict.get(target)
            if (objective == None) or (cutoff == None):
                continue
            var = variance[:, ii] + 1e-8
            ei_fitness = expected_improvement(preds[:, ii], np.where(var > 10000, 10000, var), # maximum variance allowed is 10000
                                              cutoff, minimize=objective=="minimize")
            return np.array(ei_fitness)

    def calc_probability_improvement_fitness(self, smiles_list):
        """ Calculate the multi-objective uncertainty-aware fitness. 
        If the input arguments do not contain all the tasks that chemprop has trained on, run the subset optimization or single-objective."""
        fitness_list = []
        preds, variance = self.predict(smiles_list)
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict.get(target)
            cutoff = self.target_cutoff_dict.get(target)
            if (objective == None) or (cutoff == None):
                continue
            prob_list = []
            for pred, var in zip(preds[:, ii], variance[:, ii]):
                cdf = gaussian_cdf(pred, var, cutoff)
                prob_list.append(cdf)
            prob = np.array(prob_list)
            if objective == "maximize":
                prob = 1 - prob
            fitness_list.append(prob)
        return np.array(fitness_list)
    
    def calc_overall_fitness(self, smiles_list):
        """Probability Improvement multi-objective fitness. """
        multiobjective_fitness = self.calc_probability_improvement_fitness(smiles_list)
        overall_fitness = np.prod(multiobjective_fitness, axis=0)
        return overall_fitness
    
    def calc_weighted_sum_fitness(self, smiles_list):
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            weight = self.target_weight_dict.get(target)
            objective = self.target_objective_dict.get(target)
            if objective == "maximize":
                overall_fitness += weight*preds[:, ii]
            elif objective == "minimize":
                overall_fitness += weight*preds[:, ii] * (-1)
        return overall_fitness
    
    def calc_scaler_fitness(self, smiles_list):
        """ Calculate the scores after normalized scaling. """
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict.get(target)
            mean_std_tuple = self.target_scaler_dict.get(target)
            if (objective == None) or (mean_std_tuple == None):
                continue
            else:
                mean, std = mean_std_tuple
            
            if objective == "maximize":
                overall_fitness += (preds[:, ii] - mean) / std
            elif objective == "minimize":
                overall_fitness += (preds[:, ii] - mean) / std * (-1)
        return overall_fitness
    
    def calc_utopian_distance_fitness(self, smiles_list):
        """ Only used in multi-objective fitness calculation. """
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict[target]
            utopian_point, std = self.target_utopian_dict[target]
            if objective == "maximize":
                fitness = (utopian_point - preds[:, ii]) / std
                fitness[fitness < 0] = 0
                overall_fitness += fitness
            elif objective == "minimize":
                fitness = (preds[:, ii] - utopian_point) / std
                fitness[fitness < 0] = 0
                overall_fitness += fitness
        return overall_fitness * (-1)
    
    def calc_hybrid_fitness(self, smiles_list):
        """ When the prediction is not reached utopian point, use utopian distance. Otherwise, use scaler fitness. """
        overall_utopian_fitness = self.calc_utopian_distance_fitness(smiles_list)
        overall_scaler_fitness = self.calc_scaler_fitness(smiles_list)
        mask = np.ones(overall_scaler_fitness.shape)
        mask[np.nonzero(overall_utopian_fitness)[0]] = 0
        overall_hybrid_fitness = overall_utopian_fitness + mask * overall_scaler_fitness
        return overall_hybrid_fitness

    
    def single_overall_fitness(self, single_smiles):
        """Single prediction is very slow, not recommended. """
        fitness = self.calc_overall_fitness([single_smiles])
        return fitness[0]
    
    def single_scalarization_fitness(self, single_smiles):
        fitness = self.calc_scalarization_fitness([single_smiles])
        return fitness[0]
    
    def single_scaler_fitness(self, single_smiles):
        fitness = self.calc_scaler_fitness([single_smiles])
        return fitness[0]
    
    def penalty(self, smiles):
        """All the model inherting this must include the penalty fuinction to penalize the specific SMILES."""
        return 0

    def batch_penalty(self, smiles_list):
        return np.array([self.penalty(smiles) for smiles in smiles_list])
        
    def batch_uncertainty_fitness(self, smiles_list):
        """" Probability improvement. """
        return self.calc_overall_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def batch_expected_improvement_fitness(self, smiles_list):
        """ Expected improvement. """
        return self.calc_expected_improvement_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def uncertainty_fitness(self, smiles):
        return self.single_overall_fitness(smiles) + self.penalty(smiles)

    def scalarization_fitness(self, smiles):
        return self.single_scalarization_fitness(smiles) + self.penalty(smiles)
    
    def batch_scaler_fitness(self, smiles_list):
        return self.calc_scaler_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def scaler_fitness(self, smiles):
        return self.single_scaler_fitness(smiles) + self.penalty(smiles)
    
    def batch_utopian_distance_fitness(self, smiles_list):
        return self.calc_utopian_distance_fitness(smiles_list) + self.batch_penalty(smiles_list)
    
    def batch_hybrid_fitness(self, smiles_list):
        return self.calc_hybrid_fitness(smiles_list) + self.batch_penalty(smiles_list)


class MultipleChempropUncertaintyPredictor():
    def __init__(self, model_paths: List[str], uncertainty_methods:List[str], calibration_factors:List[float]=None,
                 batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_paths = model_paths
        self.uncertainty_methods = uncertainty_methods
        self.models_list = []
        for model_path, uncertainty_method in zip(self.model_paths, self.uncertainty_methods):
            # Each model in the models_list can only be single-task model.
            self.models_list.append(ChempropUncertaintyPredictor(model_path, uncertainty_method, batch_size=batch_size, device=device))
        
    def load_cutoffs_objectives(self, cutoffs_list, objectives_list):
        assert set(objectives_list).issubset(set(["maximize", "minimize"])) 
        self.cutoffs_list = cutoffs_list
        self.objectives_list = objectives_list
        return
    
    def load_scalers_objectives(self, scalers_list, objectives_list):
        assert set(objectives_list).issubset(set(["maximize", "minimize"]))
        self.scalers_list = scalers_list
        self.objectives_list = objectives_list
        return
    
    def load_utopians_objectives(self, utopians_list, objectives_list):
        assert set(objectives_list).issubset(set(["maximize", "minimize"]))
        self.utopians_list = utopians_list
        self.objectives_list = objectives_list
        return
    
    def predict(self, smiles_list):
        means_list = []
        uncs_list = []
        for model in self.models_list:
            mean, total_uncertainty = model.predict(smiles_list)
            means_list.append(mean)
            uncs_list.append(total_uncertainty)
        # post process
        means = np.hstack(means_list)
        uncs = np.hstack(uncs_list)
        return means, uncs

    def calc_probability_improvement_fitness(self, smiles_list):
        """ Calculate the multi-objective uncertainty-aware fitness. 
        If the input arguments do not contain all the tasks that chemprop has trained on, run the subset optimization or single-objective."""
        fitness_list = []
        preds, variance = self.predict(smiles_list)
        for ii, (cutoff, objective) in enumerate(zip(self.cutoffs_list, self.objectives_list)):
            prob_list = []
            for pred, var in zip(preds[:, ii], variance[:, ii]):
                cdf = gaussian_cdf(pred, var, cutoff)
                prob_list.append(cdf)
            prob = np.array(prob_list)
            if objective == "maximize":
                prob = 1 - prob
            fitness_list.append(prob)
        return np.array(fitness_list)
    
    def calc_overall_fitness(self, smiles_list):
        """Probability Improvement multi-objective fitness. """
        multiobjective_fitness = self.calc_probability_improvement_fitness(smiles_list)
        overall_fitness = np.prod(multiobjective_fitness, axis=0)
        return overall_fitness
    
    def calc_scaler_fitness(self, smiles_list):
        """ Calculate the scores after normalized scaling. """
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, (mean_std_tuple, objective) in enumerate(zip(self.scalers_list, self.objectives_list)):
            mean, std = mean_std_tuple
            if objective == "maximize":
                overall_fitness += (preds[:, ii] - mean) / std
            elif objective == "minimize":
                overall_fitness += (preds[:, ii] - mean) / std * (-1)
        return overall_fitness
    
    def calc_utopian_distance_fitness(self, smiles_list):
        """ Only used in multi-objective fitness calculation. """
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, (utopians, objective) in enumerate(zip(self.utopians_list, self.objectives_list)):
            utopian_point, std = utopians
            if objective == "maximize":
                fitness = (utopian_point - preds[:, ii]) / std
                fitness[fitness < 0] = 0
                overall_fitness += fitness
            elif objective == "minimize":
                fitness = (preds[:, ii] - utopian_point) / std
                fitness[fitness < 0] = 0
                overall_fitness += fitness
        return overall_fitness * (-1)
    
    def calc_hybrid_fitness(self, smiles_list):
        """ When the prediction is not reached utopian point, use utopian distance. Otherwise, use scaler fitness. """
        overall_utopian_fitness = self.calc_utopian_distance_fitness(smiles_list)
        overall_scaler_fitness = self.calc_scaler_fitness(smiles_list)
        mask = np.ones(overall_scaler_fitness.shape)
        mask[np.nonzero(overall_utopian_fitness)[0]] = 0
        overall_hybrid_fitness = overall_utopian_fitness + mask * overall_scaler_fitness
        return overall_hybrid_fitness
    
    def penalty(self, smiles):
        """All the model inherting this must include the penalty fuinction to penalize the specific SMILES."""
        return 0

    def batch_penalty(self, smiles_list):
        return np.array([self.penalty(smiles) for smiles in smiles_list])
        
    def batch_uncertainty_fitness(self, smiles_list):
        """" Probability improvement. """
        return self.calc_overall_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def batch_scaler_fitness(self, smiles_list):
        return self.calc_scaler_fitness(smiles_list) + self.batch_penalty(smiles_list)
    
    def batch_utopian_distance_fitness(self, smiles_list):
        return self.calc_utopian_distance_fitness(smiles_list) + self.batch_penalty(smiles_list)
    
    def batch_hybrid_fitness(self, smiles_list):
        return self.calc_hybrid_fitness(smiles_list) + self.batch_penalty(smiles_list)
