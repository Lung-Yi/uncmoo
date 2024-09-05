from uncmoo.pred_utils import DockingScorePredictor, OrganicEmitterScorePredictor, \
                            ReactivityPredictor, DockstringPredictor
from uncmoo.args import JanusArgs, CommonArgs
import pandas as pd
from uncmoo.janus_utils import ModifiedJanus
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors
RDLogger.DisableLog("rdApp.*")

import selfies
import os
import argparse

   
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

if __name__ == "__main__":
    args = JanusArgs().parse_args()
    os.makedirs(args.result_path, exist_ok = True)
    if len(args.surrogate_model_path) == 1:
         single_model = True
    else:
         single_model = False

    if args.benchmark_dataset == "docking":
        predict_func = DockingScorePredictor
    elif args.benchmark_dataset == "organic_emitter":
        if single_model:
            predict_func = OrganicEmitterScorePredictor
        elif len(args.surrogate_model_path) == len(args.target_columns):
            model_path_dict = {}
        else:
            raise ValueError("Not implement separate multi-task model yet.")

    # elif args.benchmark_dataset in ["hce_advanced", "hce_simple"]:
    #     predict_func = HCEPredictor
    elif args.benchmark_dataset == "reactivity":
        predict_func = ReactivityPredictor
    elif args.benchmark_dataset == "dockstring":
        predict_func = DockstringPredictor
    else:
        raise ValueError("Not supporting this dataset {}.".format(args.benchmark_dataset))

    
    
    if args.fitness_method == 'uncertainty':
        assert len(args.target_columns) == len(args.target_objective) == len(args.target_cutoff)
        target_cutoff_dict = {}
        target_objective_dict = {}
        for target, objective, cutoff in zip(args.target_columns, args.target_objective, args.target_cutoff):
            target_objective_dict.update({target: objective})
            target_cutoff_dict.update({target: cutoff})
        if single_model:
            unc_model = predict_func(args.surrogate_model_path[0])
        else:
            unc_model = predict_func(model_path_dict)
        unc_model.load_target_cutoff(target_cutoff_dict, target_objective_dict)
        # scores = unc_model.calc_overall_fitness(train_data['smiles'])
        if args.batch_pred:
            fitness_function = unc_model.batch_uncertainty_fitness
        else:
            fitness_function = unc_model.uncertainty_fitness
    
    elif args.fitness_method == 'expected_improvement':
        assert len(args.target_columns) == len(args.target_objective) == len(args.target_cutoff) == 1
        target_cutoff_dict = {}
        target_objective_dict = {}
        for target, objective, cutoff in zip(args.target_columns, args.target_objective, args.target_cutoff):
            target_objective_dict.update({target: objective})
            target_cutoff_dict.update({target: cutoff})
        if single_model:
            unc_model = predict_func(args.surrogate_model_path[0])
        else:
            unc_model = predict_func(model_path_dict)
        unc_model.load_target_cutoff(target_cutoff_dict, target_objective_dict)
        if args.batch_pred:
            fitness_function = unc_model.batch_expected_improvement_fitness
        else:
            raise ValueError("Only support batch calculations.")

    elif args.fitness_method == 'scalarization':
        assert len(args.target_columns) == len(args.target_weight)
        target_weight_dict = {}
        for target, weight in zip(args.target_columns, args.target_weight):
            target_weight_dict.update({target: weight})
        if single_model:
            unc_model = predict_func(args.surrogate_model_path[0])
        else:
            unc_model = predict_func(model_path_dict)
        unc_model.load_target_weights(target_weight_dict)
        # scores = unc_model.calc_scalarization_fitness(train_data['smiles'])
        if args.batch_pred:
            fitness_function = unc_model.batch_scalarization_fitness
        else:
            fitness_function = unc_model.scalarization_fitness

    elif args.fitness_method == 'scaler':
        assert len(args.target_columns)*2 == len(args.target_objective)*2 == len(args.target_scaler)
        target_scaler_dict = {}
        target_objective_dict = {}
        for i, (target, objective) in enumerate(zip(args.target_columns, args.target_objective)):
            target_objective_dict.update({target: objective})
            target_scaler_dict.update({target: (args.target_scaler[2*i], args.target_scaler[2*i+1])})
        if single_model:
            unc_model = predict_func(args.surrogate_model_path[0])
        else:
            unc_model = predict_func(model_path_dict)
        unc_model.load_target_scaler(target_scaler_dict, target_objective_dict)
        # scores = unc_model.calc_scaler_fitness(train_data['smiles'])
        if args.batch_pred:
            fitness_function = unc_model.batch_scaler_fitness
        else:
            fitness_function = unc_model.scaler_fitness

    elif args.fitness_method == "utopian":
        assert len(args.target_columns)*2 == len(args.target_objective)*2 == len(args.target_utopian)
        target_utopian_dict = {}
        target_objective_dict = {}
        for i, (target, objective) in enumerate(zip(args.target_columns, args.target_objective)):
            target_objective_dict.update({target: objective})
            target_utopian_dict.update({target: (args.target_utopian[2*i], args.target_utopian[2*i+1])})
        if single_model:
            unc_model = predict_func(args.surrogate_model_path[0])
        else:
            unc_model = predict_func(model_path_dict)
        unc_model.load_utopian_objective(target_utopian_dict, target_objective_dict)
        if args.batch_pred:
            fitness_function = unc_model.batch_utopian_distance_fitness
        else:
            raise ValueError("Not Implement")
    elif args.fitness_method == "hybrid":
        assert len(args.target_columns)*2 == len(args.target_objective)*2 == len(args.target_utopian) == len(args.target_scaler)
        target_utopian_dict = {}
        target_objective_dict = {}
        target_scaler_dict = {}
        for i, (target, objective) in enumerate(zip(args.target_columns, args.target_objective)):
            target_objective_dict.update({target: objective})
            target_utopian_dict.update({target: (args.target_utopian[2*i], args.target_utopian[2*i+1])})
            target_scaler_dict.update({target: (args.target_scaler[2*i], args.target_scaler[2*i+1])})
        if single_model:
            unc_model = predict_func(args.surrogate_model_path[0])
        else:
            unc_model = predict_func(model_path_dict)
        unc_model.load_utopian_objective(target_utopian_dict, target_objective_dict)
        unc_model.load_target_scaler(target_scaler_dict, target_objective_dict)
        if args.batch_pred:
            fitness_function = unc_model.batch_hybrid_fitness
        else:
            raise ValueError("Not Implement")

    else:
        raise ValueError("Not supporting the fitness method: {}".format(args.fitness_method))
    
    top_data = pd.read_csv(args.start_smiles_path)
    if len(target_objective_dict) != 1:
        top_data = top_data.sort_values(by='normalized_scores', ascending=False)
    else:
        objective = args.target_objective[0]
        if objective == "maximize":
            top_data = top_data.sort_values(by=args.target_columns[0], ascending=False)
        elif objective == "minimize":
            top_data = top_data.sort_values(by=args.target_columns[0], ascending=True)

    top_data = top_data[["smiles"]]
    top_data = top_data[:args.n_sample]
    sample_data_file = "start_smiles_top_{}".format(args.n_sample) + ".csv"
    sample_data_path = os.path.join(args.result_path, sample_data_file)
    top_data.to_csv(sample_data_path, index=False)
    del top_data

    # # Set your SELFIES constraints (below used for manuscript)
    # default_constraints = selfies.get_semantic_constraints()
    # new_constraints = default_constraints
    # new_constraints['S'] = 2
    # new_constraints['P'] = 3
    # selfies.set_semantic_constraints(new_constraints)  # update constraints

    if args.alphabet_path:
        with open(args.alphabet_path, "r") as z:
            alphabet = [alpha.strip() for alpha in z.readlines()]
    else:
        alphabet = None

    common_args = CommonArgs().parse_args()
    common_args = namespace_to_dict(common_args)
    algorithm_args =  {k: v for k, v in namespace_to_dict(args).items() if k not in common_args.keys()}

    print(fitness_function)
    # Create JANUS object.
    agent = ModifiedJanus(
        work_dir = args.result_path,            # where the results are saved
        fitness_function = fitness_function,    # user-defined fitness for given smiles
        start_population = sample_data_path,    # file with starting smiles population
        batch_pred = args.batch_pred,
        alphabet = alphabet,
        **algorithm_args
    )
    agent.run()     # RUN IT!