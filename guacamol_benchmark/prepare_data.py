import pandas as pd
from rdkit import Chem
import random
import argparse
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors, AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from multiprocessing import Pool

def logP(mol) -> float:
    return Descriptors.MolLogP(mol)
def tpsa(mol) -> float:
    return Descriptors.TPSA(mol)

class similarityCalculatir():
    def __init__(self):
        self.mol_Aripiprazole = Chem.MolFromSmiles("Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl")
        self.mol_Albuterol = Chem.MolFromSmiles("CC(C)(C)NCC(O)c1ccc(O)c(CO)c1")
        self.mol_Mestranol = Chem.MolFromSmiles("C#CC1(O)CCC2C3CCc4cc(OC)ccc4C3CCC21C")
        self.mol_Tadalafil  = Chem.MolFromSmiles("O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C")
        self.mol_Sildenafil = Chem.MolFromSmiles("CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")
        self.mol_Camphor = Chem.MolFromSmiles("CC1(C)C2CCC1(C)C(=O)C2")
        self.mol_Menthol = Chem.MolFromSmiles("CC(C)C1CCC(C)CC1O")
        self.mol_Fexofenadine = Chem.MolFromSmiles("CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
        self.mol_Ranolazine = Chem.MolFromSmiles("COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2")

        self.fp_Aripiprazole = AllChem.GetMorganFingerprint(self.mol_Aripiprazole, 2)
        self.fp_Albuterol = AllChem.GetMorganFingerprint(self.mol_Albuterol, 2, useFeatures=True)
        self.fp_Mestranol = AllChem.GetAtomPairFingerprint(self.mol_Mestranol, maxLength=10)
        self.fp_Tadalafil = AllChem.GetMorganFingerprint(self.mol_Tadalafil, 3)
        self.fp_Sildenafil = AllChem.GetMorganFingerprint(self.mol_Sildenafil, 3)
        self.fp_Camphor = AllChem.GetMorganFingerprint(self.mol_Camphor, 2)
        self.fp_Menthol = AllChem.GetMorganFingerprint(self.mol_Menthol, 2)
        self.fp_Fexofenadine = AllChem.GetAtomPairFingerprint(self.mol_Fexofenadine, maxLength=10)
        self.fp_Ranolazine = AllChem.GetAtomPairFingerprint(self.mol_Ranolazine, maxLength=10)

    def similarity_Aripiprazole(self, mol):
        fp = AllChem.GetMorganFingerprint(mol, 2)
        return TanimotoSimilarity(fp, self.fp_Aripiprazole)
    def similarity_Albuterol(self, mol):
        fp = AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)
        return TanimotoSimilarity(fp, self.fp_Albuterol)
    def similarity_Mestranol(self, mol):
        fp = AllChem.GetAtomPairFingerprint(mol, maxLength=10)
        return TanimotoSimilarity(fp, self.fp_Mestranol)
    def similarity_Tadalafil(self, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3)
        return TanimotoSimilarity(fp, self.fp_Tadalafil)
    def similarity_Sildenafil(self, mol):
        fp = AllChem.GetMorganFingerprint(mol, 3)
        return TanimotoSimilarity(fp, self.fp_Sildenafil)
    def similarity_Camphor(self, mol):
        fp = AllChem.GetMorganFingerprint(mol, 2)
        return TanimotoSimilarity(fp, self.fp_Camphor)
    def similarity_Menthol(self, mol):
        fp = AllChem.GetMorganFingerprint(mol, 2)
        return TanimotoSimilarity(fp, self.fp_Menthol)
    def similarity_Fexofenadine(self, mol):
        fp = AllChem.GetAtomPairFingerprint(mol, maxLength=10)
        return TanimotoSimilarity(fp, self.fp_Fexofenadine)
    def similarity_Ranolazine(self, mol):
        fp = AllChem.GetAtomPairFingerprint(mol, maxLength=10)
        return TanimotoSimilarity(fp, self.fp_Ranolazine)
    
def convert_smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_fraction', type=float, default=0.001)
    parser.add_argument('--num_cpus', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.random_seed)

    sim = similarityCalculatir()
    function_dict = {"logP": logP, "tpsa": tpsa, "similarity_Aripiprazole": sim.similarity_Aripiprazole,
                    "similarity_Albuterol": sim.similarity_Albuterol, "similarity_Mestranol": sim.similarity_Mestranol,
                    "similarity_Tadalafil": sim.similarity_Tadalafil, "similarity_Sildenafil": sim.similarity_Sildenafil,
                    "similarity_Camphor": sim.similarity_Camphor, "similarity_Menthol": sim.similarity_Menthol,
                    "similarity_Menthol": sim.similarity_Menthol, "similarity_Fexofenadine": sim.similarity_Fexofenadine,
                    "similarity_Ranolazine": sim.similarity_Ranolazine}
    
    def multiprocess_mol_list(smiles_list, num_cpus=32):
        with Pool(processes=num_cpus) as pool:
            mol_list = pool.map(convert_smiles_to_mol, smiles_list)
        return mol_list

    def multiprocess_calculate(mol_list: list, func_name: str, num_cpus=32) -> list:
        func = function_dict[func_name]
        with Pool(processes=num_cpus) as pool:
            results = pool.map(func, mol_list)
        return results

    data_files = {"train": "guacamol_v1_train.smiles", 
                "valid": "guacamol_v1_valid.smiles",
                "test": "guacamol_v1_test.smiles"}
    
    for split_type, file_name in data_files.items():
        with open(f"guacamol_dataset/{file_name}", "r") as f:
            data = f.readlines()
        data = [line.strip() for line in data]
        if split_type == "train":
            data_sampled = random.sample(data, 10000)
            data_sampled = sorted(data_sampled, key=lambda x: data.index(x))
        elif split_type == "valid":
            data_sampled = random.sample(data, 2000)
            data_sampled = sorted(data_sampled, key=lambda x: data.index(x))
        elif split_type == "test":
            data_sampled = random.sample(data, 10000)
            data_sampled = sorted(data_sampled, key=lambda x: data.index(x))
        
        mol_data = multiprocess_mol_list(data_sampled, args.num_cpus)

        pd_data = pd.DataFrame()
        pd_data["smiles"] = data_sampled
        for func_name in function_dict.keys():
            properties = multiprocess_calculate(mol_data, func_name, args.num_cpus)
            pd_data[func_name] = properties
        pd_data.to_csv(f"guacamol_dataset/guacamol_sample_{split_type}.csv",float_format="%.4f",index=False)

