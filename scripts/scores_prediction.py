import os
import pickle
import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from uncmoo.pred_utils import DockingScorePredictor, OrganicEmitterScorePredictor, HCEPredictor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["docking", "docking_all", "organic_emitter", "hce_advanced", "hce_simple", "hce_all"])
    args = parser.parse_args()

    if args.dataset in ["docking", "docking_all"]:
        column_names = ["1syh score", "4lde score", "6y2f score"]
        model_path = "../chemprop_unc/save_models/docking/try_80/fold_0/model_0/model.pt"
        unc_model = DockingScorePredictor(model_path)
    elif args.dataset == "organic_emitter":
        column_names = ["singlet-triplet value", "oscillator strength", "multi-objective value"]
        model_path = "../chemprop_unc/save_models/organic_emitter/ensemble_mve/fold_0"
        unc_model = OrganicEmitterScorePredictor(model_path)
    # elif args.dataset == "hce_advanced":
    #     column_names = ["pce_pcbm_sas", "pce_pcdtbt_sas"]
    # elif args.dataset == "hce_simple":
    #     column_names = ["dipm", "gap", "lumo", "combined"]
    # elif args.dataset == "hce_all":
    #     column_names = ["dipm", "gap", "lumo", "combined", "pce_pcbm_sas", "pce_pcdtbt_sas"]
    else:
        raise ValueError("Not implement the surrogate model function:", args.dataset)

    # separate_path = "../RESULTS/{}/separate".format(args.dataset)
    save_path = "../RESULTS/{}".format(args.dataset)
    # try:
    with open(os.path.join(save_path, "cal_dict.pkl"), "rb") as f:
        reference_dict = pickle.load(f)
    # except:
        # save_dict = dict()

    # files = [os.path.join(separate_path, f) for f in os.listdir(separate_path) if os.path.isfile(os.path.join(separate_path, f))]
    predict_dict = dict()
    smiles_list = list(reference_dict.keys())
    preds, all_unc = unc_model.predict(smiles_list)
    pred_df = pd.DataFrame()
    for smiles, sub_preds in zip(smiles_list, preds):
        predict_dict.update({smiles: tuple(sub_preds)})
        df_dict = {"smiles":smiles}
        df_dict.update({column_names[i]:[sub_preds[i]] for i in range(len(column_names))})
        pred_df = pd.concat([pred_df, pd.DataFrame(df_dict)], axis=0)

    # print(pred_df)
    pred_df = pred_df.sort_values(by="smiles")
    pred_df.to_csv(os.path.join(save_path, "predict_results.csv"), index=False)
    reference_df = pd.read_csv(os.path.join(save_path, "cal_results.csv"))
    reference_df = reference_df.sort_values(by="smiles")
    reference_df.to_csv(os.path.join(save_path, "cal_results.csv"), index=False)
    with open(os.path.join(save_path, "predict_dict.pkl"), "wb") as f:
        pickle.dump(predict_dict, f)

    

