import os
import pickle
import argparse
import pandas as pd


if __name__ == '__main__':


    parent_dir = "../RESULTS/docking_all"
    data_path = "../Tartarus/datasets/docking.csv"
    data = pd.read_csv(data_path)

    total_smiles_list = list(data["smiles"])
    total_smiles_set = set(total_smiles_list)
    previous_file_path = os.path.join(parent_dir, "cal_dict.pkl")
    if os.path.exists(previous_file_path):
        with open(previous_file_path, "rb") as g:
            save_dict = pickle.load(g)
        total_smiles_set = total_smiles_set - set(save_dict.keys())
    else:
        save_dict = dict()

    
    total_smiles_list = list(total_smiles_set)
    total_smiles_list = sorted(total_smiles_list, key = lambda x : len(x))
    print(total_smiles_list)


    os.makedirs("../job_uncmoo/{}/template".format("docking_all"), exist_ok=True)
    with open("single_score_calc_template.sh", "r") as f:
        template = f.read()

    for i, smiles in enumerate(total_smiles_list):
        new_template = template.replace("$$$DATASET$$$", "docking_all")
        new_template = new_template.replace("$$$SMILES$$$", smiles)
        with open("../job_uncmoo/{}/template/score_{}.sh".format("docking_all", i+1), "w") as z:
            z.write(new_template)

        

