import os
import pickle
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["docking", "docking_all", "organic_emitter", "hce_advanced", "hce_simple", "hce_all"])
    args = parser.parse_args()

    if args.dataset in ["docking", "docking_all"]:
        column_names = ["1syh score", "4lde score", "6y2f score"]
    elif args.dataset == "organic_emitter":
        column_names = ["singlet-triplet value", "oscillator strength", "multi-objective value"]
    elif args.dataset == "hce_advanced":
        column_names = ["pce_pcbm_sas", "pce_pcdtbt_sas"]
    elif args.dataset == "hce_simple":
        column_names = ["dipm", "gap", "lumo", "combined"]
    elif args.dataset == "hce_all":
        column_names = ["dipm", "gap", "lumo", "combined", "pce_pcbm_sas", "pce_pcdtbt_sas"]
    else:
        raise ValueError("Not implement the calculation function:", args.dataset)

    separate_path = "../RESULTS/{}/separate".format(args.dataset)
    save_path = "../RESULTS/{}".format(args.dataset)
    try:
        with open(os.path.join(save_path, "cal_dict.pkl"), "rb") as f:
            save_dict = pickle.load(f)
    except:
        save_dict = dict()

    files = [os.path.join(separate_path, f) for f in os.listdir(separate_path) if os.path.isfile(os.path.join(separate_path, f))]

    
    for filename in files:    
        df = pd.read_csv(filename)
        smiles = df['smiles'][0]
        values = tuple(df[column_names].values[0])
        save_dict.update({smiles: values})
        # all_data = pd.concat([all_data, df], ignore_index=True)
    all_data = pd.DataFrame()
    for key, values in save_dict.items():
        all_data = pd.concat([all_data, pd.DataFrame([[key]+list(values)], columns=["smiles"]+column_names)],axis=0,ignore_index=True)
    print(all_data)

    # save the files:
    with open(os.path.join(save_path, "cal_dict.pkl"), "wb") as f:
        pickle.dump(save_dict, f)
    all_data.to_csv(os.path.join(save_path, "cal_results.csv"), index=False)
    

    

