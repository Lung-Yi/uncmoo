import os
from tartarus import docking, tadf, pce
import pandas as pd
import pickle
import argparse

def hce_advanced_scores(smiles):
    _, _, _, _, pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smiles)
    return pce_pcbm_sas, pce_pcdtbt_sas

def hce_simple_scores(smiles):
    dipm, gap, lumo, combined, _, _ = pce.get_properties(smiles)
    return dipm, gap, lumo, combined

def hce_all_scores(smiles):
    dipm, gap, lumo, combined, pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smiles)
    return dipm, gap, lumo, combined, pce_pcbm_sas, pce_pcdtbt_sas

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
    parser.add_argument('--smiles', type=str)
    parser.add_argument('--dataset', type=str, choices=["docking", "docking_all","organic_emitter", "hce_advanced", "hce_simple", "hce_all"])
    args = parser.parse_args()

    if args.dataset in ["docking", "docking_all"]:
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
    elif args.dataset == "hce_all":
        calc_function = hce_all_scores
        column_names = ["dipm", "gap", "lumo", "combined", "pce_pcbm_sas", "pce_pcdtbt_sas"]
    else:
        raise ValueError("Not implement the calculation function:", args.dataset)


    parent_dir = os.path.join("../RESULTS", args.dataset)
    separate_save_dir = os.path.join(parent_dir, "separate")
    os.makedirs(separate_save_dir, exist_ok=True)

    scores = calc_function(args.smiles)
    print(args.smiles, scores)
    df = pd.DataFrame([[args.smiles]+[*scores]], columns=["smiles"]+column_names)
    df.to_csv(os.path.join(separate_save_dir,"{}.csv".format(hash(args.smiles))),index=False)
