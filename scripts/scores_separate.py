import os
import pickle
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dataset', type=str, choices=["docking", "organic_emitter", "hce_advanced", "hce_simple", "reactivity"])
    args = parser.parse_args()

    if args.dataset == "docking":
        parent_dir_list = ["docking_1syh", "docking_4lde", "docking_6y2f"]
        parent_dir_list = [os.path.join("../RESULTS", x) for x in parent_dir_list]
    elif args.dataset == "organic_emitter":
        parent_dir_list = ["organic_emitter", "organic_emitter_os", "organic_emitter_stv"]
        parent_dir_list = [os.path.join("../RESULTS", x) for x in parent_dir_list]
    elif args.dataset == "reactivity":
        parent_dir_list = ["reactivity", "reactivity_activation_energy", "reactivity_reaction_energy"]
        parent_dir_list = [os.path.join("../RESULTS", x) for x in parent_dir_list]
    else:
        parent_dir = os.path.join("../RESULTS", args.dataset)
        parent_dir_list = [parent_dir]

    sub_dir_name = ["janus_scaler", "janus_uncertainty", "janus_uncertainty_cutoff", "janus_utopian", "janus_utopian_cutoff", "janus_hybrid"]
    # ["janus_scaler", "janus_uncertainty", "janus_uncertainty_cutoff", "janus_utopian", "janus_utopian_cutoff", "janus_hybrid"]
    data_dir = []

    for parent_dir in parent_dir_list:
        for sub_dir in sub_dir_name:
            new_dir = [os.path.join(parent_dir, sub_dir+"_{}".format(fold)) for fold in range(1, args.fold+1)]
            data_dir += new_dir
    total_smiles_list = []
    for sub_dir in data_dir:
        # explore SMILES
        filename = os.path.join(sub_dir, "population_explore.txt")
        try:
            with open(filename, "r") as f:
                data = f.read()
        except:
            continue
        line = data.strip()
        total_smiles_list += line.split(' ')[:20]
        # local search SMILES
        filename = os.path.join(sub_dir, "population_local_search.txt")
        try:
            with open(filename, "r") as f:
                data = f.read()
        except:
            continue
        line = data.strip()
        total_smiles_list += line.split(' ')[:20]
        # best within each generation step
        filename = os.path.join(sub_dir, "generation_all_best.txt")
        smiles_list = []
        try:
            with open(filename, "r") as f:
                data = f.readlines()
        except:
            continue
        for line in data:
            smiles_list.append(line.split(", ")[1])
        total_smiles_list += smiles_list  
      

    total_smiles_set = set(total_smiles_list)
    # load the file that has been previously calculated
    parent_dir = os.path.join("../RESULTS", args.dataset)
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

        

