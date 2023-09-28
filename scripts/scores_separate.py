import os
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dataset', type=str, choices=["docking", "organic_emitter", "hce_advanced", "hce_simple"])
    args = parser.parse_args()


    parent_dir = os.path.join("../RESULTS", args.dataset)
    

    sub_dir_name = ["janus_scaler", "janus_uncertainty", "janus_utopian"]
    data_dir = []

    for sub_dir in sub_dir_name:
        new_dir = [os.path.join(parent_dir, sub_dir+"_{}".format(fold)) for fold in range(1, args.fold+1)]
        data_dir += new_dir

    total_smiles_list = []
    for sub_dir in data_dir:
        # filename = os.path.join(sub_dir, "generation_all_best.txt")
        filename = os.path.join(sub_dir, "population_explore.txt")
        try:
            with open(filename, "r") as f:
                # data = f.readlines()
                data = f.read()
        except:
            continue
        
        # for line in data:
            # smiles = line.split(", ")[1]
            # total_smiles_list.append(smiles)
        line = data.strip()
        total_smiles_list += line.split(' ')[:50]

    total_smiles_set = set(total_smiles_list)
    # load the file that has been previously calculated
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


    os.makedirs("../job_uncmoo/{}/calc_scores/template".format(args.dataset), exist_ok=True)
    with open("single_score_calc_template.sh", "r") as f:
        template = f.read()

    for i, smiles in enumerate(total_smiles_list):
        new_template = template.replace("$$$DATASET$$$", args.dataset)
        new_template = new_template.replace("$$$SMILES$$$", smiles)
        with open("../job_uncmoo/{}/calc_scores/template/score_{}.sh".format(args.dataset, i+1), "w") as z:
            z.write(new_template)

        

