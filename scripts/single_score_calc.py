import os
import pandas as pd
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str)
    parser.add_argument('--dataset', type=str, choices=["docking", "docking_all","organic_emitter", "hce_advanced", "hce_simple", "hce_all", "reactivity", "reactivity_all", "organic_emitter_all"])
    args = parser.parse_args()

    if args.dataset in ["docking", "docking_all"]:
        from tartarus import docking
        def docking_scores(smiles):
            A = docking.get_score(smiles, docking_target='1syh')
            B = docking.get_score(smiles, docking_target='4lde')
            C = docking.get_score(smiles, docking_target='6y2f')
            return A, B, C
        calc_function = docking_scores
        column_names = ["1syh score", "4lde score", "6y2f score"]

    elif args.dataset in ["organic_emitter", "organic_emitter_all"]:
        from tartarus import tadf
        def organic_emitter_scores(smiles):
            try:
                st, osc, combined = tadf.get_properties(smiles)#, scratch="/home/lungyi/tmp")
            except:
                st, osc, combined = -10000, -10000, -10000
            if st == -10000:
                return st, osc, 10000 # the goal is to minimize the abs_diff_vee
            else:
                return -st, osc, (combined + st - osc)*(-1) # return the real abs_diff_vee
        calc_function = organic_emitter_scores
        column_names = ["singlet-triplet value", "oscillator strength", "abs_diff_vee"]

    elif args.dataset == "hce_advanced":
        from tartarus import pce
        def hce_advanced_scores(smiles):
            _, _, _, _, pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smiles)
            return pce_pcbm_sas, pce_pcdtbt_sas
        calc_function = hce_advanced_scores
        column_names = ["pce_pcbm_sas", "pce_pcdtbt_sas"]
    elif args.dataset == "hce_simple":
        from tartarus import pce
        def hce_simple_scores(smiles):
            dipm, gap, lumo, combined, _, _ = pce.get_properties(smiles)
            return dipm, gap, lumo, combined
        calc_function = hce_simple_scores
        column_names = ["dipm", "gap", "lumo", "combined"]
    elif args.dataset == "hce_all":
        from tartarus import pce
        def hce_all_scores(smiles):
            dipm, gap, lumo, combined, pce_pcbm_sas, pce_pcdtbt_sas = pce.get_properties(smiles)
            return dipm, gap, lumo, combined, pce_pcbm_sas, pce_pcdtbt_sas
        calc_function = hce_all_scores
        column_names = ["dipm", "gap", "lumo", "combined", "pce_pcbm_sas", "pce_pcdtbt_sas"]
    elif args.dataset in ["reactivity", "reactivity_all"]:
        from tartarus import reactivity
        def reactivity_scores(smiles):
            Ea, Er, sum_Ea_Er, diff_Ea_Er  = reactivity.get_properties(smiles, n_procs=1)
            return Ea, Er
        calc_function = reactivity_scores
        column_names = ["Ea", "Er"]
    else:
        raise ValueError("Not implement the calculation function:", args.dataset)


    parent_dir = os.path.join("../RESULTS", args.dataset)
    separate_save_dir = os.path.join(parent_dir, "separate")
    os.makedirs(separate_save_dir, exist_ok=True)

    scores = calc_function(args.smiles)
    print(args.smiles, scores)
    df = pd.DataFrame([[args.smiles]+[*scores]], columns=["smiles"]+column_names)
    df.to_csv(os.path.join(separate_save_dir,"{}.csv".format(hash(args.smiles))),index=False)
