import os
from tartarus import docking, tadf, pce
import pandas as pd
import multiprocessing
import pickle
import argparse

def hce_advanced_scores(smiles):
    _, _, _, _, pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smiles)
    return pce_pcbm_sas, pce_pcdtbt_sas

def hce_simple_scores(smiles):
    dipm, gap, lumo, combined, _, _ = pce.get_properties(smiles)
    return dipm, gap, lumo, combined

def docking_scores(smiles):
    A = docking.get_score(smiles, docking_target='1syh')
    B = docking.get_score(smiles, docking_target='4lde')
    C = docking.get_score(smiles, docking_target='6y2f')
    return A, B, C

def organic_emitter_scores(smiles):
    try:
        st, osc, combined = tadf.get_properties(smiles)#, scratch="/home/lungyi/tmp")
    except:
        st, osc, combined = -10000, -10000, -10000
    return st, osc, combined

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dataset', type=str, choices=["docking", "organic_emitter", "hce_advanced", "hce_simple"])
    args = parser.parse_args()

    if args.dataset == "docking":
        calc_function = docking_scores
        column_names = ["1syh score", "4lde score", "6y2f score"]
    elif args.dataset == "organic_emitter":
        calc_function = organic_emitter_scores
        column_names = ["singlet-triplet value", "oscillator strength", "multi-objective value"]
    elif args.dataset == "hce_advanced":
        calc_function = hce_advanced_scores
        column_names = ["pce_pcbm_sas", "pce_pcdtbt_sas"]
    elif args.dataset == "hce_simple":
        calc_function = hce_simple_scores
        column_names = ["dipm", "gap", "lumo", "combined"]
    else:
        raise ValueError("Not implement the calculation function:", args.dataset)

    parent_dir = os.path.join("../RESULTS", args.dataset)
    sub_dir_name = ["janus_scaler", "janus_uncertainty"]#, "janus_uncertainty_tight"]
    data_dir = []

    for sub_dir in sub_dir_name:
        new_dir = [os.path.join(parent_dir, sub_dir+"_{}".format(fold)) for fold in range(1, args.fold+1)]
        data_dir += new_dir

    total_smiles_list = []
    for sub_dir in data_dir:
        filename = os.path.join(sub_dir, "generation_all_best.txt")
        try:
            with open(filename, "r") as f:
                data = f.readlines()
        except:
            continue
        for line in data:
            smiles = line.split(", ")[1]
            total_smiles_list.append(smiles)

    total_smiles_set = set(total_smiles_list)
    # load the file that has been previously calculated
    previous_file_path = os.path.join(parent_dir, "cal_dict.pkl")
    if os.path.exists(previous_file_path):
        with open(previous_file_path, "rb") as g:
            save_dict = pickle.load(g)
        total_smiles_set = total_smiles_set - set(save_dict.keys())
        save_data = pd.read_csv(os.path.join(parent_dir, "cal_results.csv"))
    else:
        save_dict = dict()
        save_data = pd.DataFrame(columns=["smiles"]+column_names)

    
    total_smiles_list = list(total_smiles_set)
    total_smiles_list = sorted(total_smiles_list, key = lambda x : len(x))
    print(total_smiles_list)

    for smiles in total_smiles_list:
        scores = calc_function(smiles)
        print(smiles, scores)
        save_dict.update({smiles: scores})
        save_data = pd.concat([save_data, pd.DataFrame([[smiles]+[*scores]], columns=save_data.columns)], ignore_index=True,axis=0)
        save_data.to_csv(os.path.join(parent_dir, "cal_results.csv"), index=False)
        with open(os.path.join(parent_dir, "cal_dict.pkl"), "wb") as g:
            pickle.dump(save_dict, g)


    # print("CPU available:")
    # print(multiprocessing.cpu_count())
    # pool = multiprocessing.Pool()
    # results = pool.map(calc_function, total_smiles_list)
    # pool.close()
    # pool.join()

    # # save the results
    # for smiles, result in zip(total_smiles_list, results):
    #     save_dict.update({smiles: result})
    #     print(smiles, result)
    #     save_data = pd.concat([save_data, pd.DataFrame([[smiles]+[*result]], columns=save_data.columns)], ignore_index=True,axis=0)

    # print(save_data)
    # save_data.to_csv(os.path.join(parent_dir, "cal_results.csv"), index=False)
    # with open(os.path.join(parent_dir, "cal_dict.pkl"), "wb") as g:
    #     pickle.dump(save_dict, g)
