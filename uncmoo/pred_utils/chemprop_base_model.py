import numpy as np
import torch
import tempfile
from abc import ABC, abstractmethod

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

class ChempropEvidentialUncertaintyPredictor(ABC):
    def __init__(self, model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.train_args, self.model, self.scalers, self.num_tasks, self.task_names = self.load_single_model(self.model_path, self.device)
        self.scaler = self.scalers[0]
        self.features_generator = []
        
    def load_single_model(self, model_path, device):
        train_args = load_args(model_path)
        num_tasks, task_names = train_args.num_tasks, train_args.task_names
        model = load_checkpoint(model_path, device=device)
        scalers = load_scalers(model_path)
        return train_args, model, scalers, num_tasks, task_names

    def predict(self, smiles_list):
        test_data = get_data_from_smiles(
        smiles=[[s] for s in smiles_list],
        skip_invalid_smiles=False,
        features_generator=self.features_generator)
        test_data = MoleculeDataset(test_data)
        test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=self.batch_size)

        gamma, mu, alpha, beta = predict(self.model, test_data_loader, scaler=self.scaler, disable_progress_bar=True, return_unc_parameters=True)
        gamma, mu, alpha, beta = np.array([gamma, mu, alpha, beta])
        total_uncertainty = beta * (1 + 1 / mu) / (alpha-1)
        return gamma, total_uncertainty
    
    def predict_allunc(self, smiles_list):
        test_data = get_data_from_smiles(
        smiles=[[s] for s in smiles_list],
        skip_invalid_smiles=False,
        features_generator=self.features_generator)
        test_data = MoleculeDataset(test_data)
        test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=self.batch_size)

        gamma, mu, alpha, beta = predict(self.model, test_data_loader, scaler=self.scaler, disable_progress_bar=True, return_unc_parameters=True)
        gamma, mu, alpha, beta = np.array([gamma, mu, alpha, beta])
        epi_uncertainty = beta / (mu*(alpha-1))
        ale_uncertainty = beta / (alpha-1)
        return gamma, ale_uncertainty, epi_uncertainty
    
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

    def calc_multiobjective_fitness(self, smiles_list):
        fitness_list = []
        preds, variance = self.predict(smiles_list)
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict[target]
            cutoff = self.target_cutoff_dict[target]
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
        multiobjective_fitness = self.calc_multiobjective_fitness(smiles_list)
        overall_fitness = np.prod(multiobjective_fitness, axis=0)
        return overall_fitness
    
    def calc_scalarization_fitness(self, smiles_list):
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            weight = self.target_weight_dict[target]
            overall_fitness += weight*preds[:, ii]
        return overall_fitness
    
    def calc_scaler_fitness(self, smiles_list):
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict[target]
            mean, std = self.target_scaler_dict[target]
            if objective == "maximize":
                overall_fitness += (preds[:, ii] - mean) / std
            elif objective == "minimize":
                overall_fitness += (preds[:, ii] - mean) / std * (-1)
        return overall_fitness
    
    def calc_utopian_distance_fitness(self, smiles_list):
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict[target]
            utopian_point, std = self.target_utopian_dict[target]
            if objective == "maximize":
                fitness = (utopian_point - preds[:, ii]) / std
                fitness[fitness < 0] = 0
                overall_fitness += fitness**2
            elif objective == "minimize":
                fitness = (preds[:, ii] - utopian_point) / std
                fitness[fitness < 0] = 0
                overall_fitness += fitness**2
        return np.sqrt(overall_fitness) * (-1)
    
    def single_overall_fitness(self, single_smiles):
        fitness = self.calc_overall_fitness([single_smiles])
        return fitness[0]
    
    def single_scalarization_fitness(self, single_smiles):
        fitness = self.calc_scalarization_fitness([single_smiles])
        return fitness[0]
    
    def single_scaler_fitness(self, single_smiles):
        fitness = self.calc_scaler_fitness([single_smiles])
        return fitness[0]
    
    @abstractmethod
    def penalty(self, smiles):
        """All the model inherting this must include the penalty fuinction to penalize the specific SMILES."""
        return 0

    def batch_penalty(self, smiles_list):
        return np.array([self.penalty(smiles) for smiles in smiles_list])
    
    def batch_uncertainty_fitness(self, smiles_list):
        return self.calc_overall_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def uncertainty_fitness(self, smiles):
        return self.single_overall_fitness(smiles) + self.penalty(smiles)
    
    def batch_scalarization_fitness(self, smiles_list):
        return self.calc_scalarization_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def scalarization_fitness(self, smiles):
        return self.single_scalarization_fitness(smiles) + self.penalty(smiles)

    def batch_scaler_fitness(self, smiles_list):
        return self.calc_scaler_fitness(smiles_list) + self.batch_penalty(smiles_list)
    
    def scaler_fitness(self, smiles):
        return self.single_scaler_fitness(smiles) + self.penalty(smiles)
    
    def batch_utopian_distance_fitness(self, smiles_list):
        return self.calc_utopian_distance_fitness(smiles_list) + self.batch_penalty(smiles_list)


class ChempropEnsembleMVEPredictor(ABC):
    def __init__(self, model_path, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        arguments = [
            '--test_path', None,
            '--preds_path', tempfile.NamedTemporaryFile().name,
            '--checkpoint_dir', model_path,
            '--uncertainty_method', "mve",
        ]
        args = PredictArgs().parse_args(arguments)

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.args, self.train_args, self.models, self.scalers, self.num_tasks, self.task_names = load_model(args)
        for model in self.models:
            model = model.to(self.device)
        self.args.num_workers = 0

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
        return np.array(mean), np.array(total_uncertainty)
    
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

    def calc_multiobjective_fitness(self, smiles_list):
        fitness_list = []
        preds, variance = self.predict(smiles_list)
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict[target]
            cutoff = self.target_cutoff_dict[target]
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
        multiobjective_fitness = self.calc_multiobjective_fitness(smiles_list)
        overall_fitness = np.prod(multiobjective_fitness, axis=0)
        return overall_fitness
    
    def calc_scalarization_fitness(self, smiles_list):
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            weight = self.target_weight_dict[target]
            overall_fitness += weight*preds[:, ii]
        return overall_fitness
    
    def calc_scaler_fitness(self, smiles_list):
        preds, _ = self.predict(smiles_list)
        overall_fitness = 0
        for ii, target in enumerate(self.task_names):
            objective = self.target_objective_dict[target]
            mean, std = self.target_scaler_dict[target]
            if objective == "maximize":
                overall_fitness += (preds[:, ii] - mean) / std
            elif objective == "minimize":
                overall_fitness += (preds[:, ii] - mean) / std * (-1)
        return overall_fitness
    
    def calc_utopian_distance_fitness(self, smiles_list):
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
    
    def single_overall_fitness(self, single_smiles):
        fitness = self.calc_overall_fitness([single_smiles])
        return fitness[0]
    
    def single_scalarization_fitness(self, single_smiles):
        fitness = self.calc_scalarization_fitness([single_smiles])
        return fitness[0]
    
    def single_scaler_fitness(self, single_smiles):
        fitness = self.calc_scaler_fitness([single_smiles])
        return fitness[0]
    
    @abstractmethod
    def penalty(self, smiles):
        """All the model inherting this must include the penalty fuinction to penalize the specific SMILES."""
        return 0

    def batch_penalty(self, smiles_list):
        return np.array([self.penalty(smiles) for smiles in smiles_list])
        
    def batch_uncertainty_fitness(self, smiles_list):
        return self.calc_overall_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def uncertainty_fitness(self, smiles):
        return self.single_overall_fitness(smiles) + self.penalty(smiles)
    
    def batch_scalarization_fitness(self, smiles_list):
        return self.calc_scalarization_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def scalarization_fitness(self, smiles):
        return self.single_scalarization_fitness(smiles) + self.penalty(smiles)
    
    def batch_scaler_fitness(self, smiles_list):
        return self.calc_scaler_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def scaler_fitness(self, smiles):
        return self.single_scaler_fitness(smiles) + self.penalty(smiles)
    
    def batch_utopian_distance_fitness(self, smiles_list):
        return self.calc_utopian_distance_fitness(smiles_list) + self.batch_penalty(smiles_list)


class ChempropUncertaintyPredictor(ABC):
    def __init__(self, model_path, uncertainty_method, batch_size=2048, device="cuda" if torch.cuda.is_available() else "cpu"):
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
        return np.array(mean), np.array(total_uncertainty)
    
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

    def calc_multiobjective_fitness(self, smiles_list):
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
        multiobjective_fitness = self.calc_multiobjective_fitness(smiles_list)
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
    
    @abstractmethod
    def penalty(self, smiles):
        """All the model inherting this must include the penalty fuinction to penalize the specific SMILES."""
        return 0

    def batch_penalty(self, smiles_list):
        return np.array([self.penalty(smiles) for smiles in smiles_list])
        
    def batch_uncertainty_fitness(self, smiles_list):
        return self.calc_overall_fitness(smiles_list) + self.batch_penalty(smiles_list)

    def uncertainty_fitness(self, smiles):
        return self.single_overall_fitness(smiles) + self.penalty(smiles)
    
    def batch_scalarization_fitness(self, smiles_list):
        return self.calc_scalarization_fitness(smiles_list) + self.batch_penalty(smiles_list)

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