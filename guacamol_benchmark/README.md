# Prepare the Guacamol benchmark dataset
The optimization benchmark is referenced in the paper: [`GuacaMol: Benchmarking Models for de Novo Molecular Design`](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839).

Please download the post-processed SMILES data in the following link [`GuacaMol dataset`](https://figshare.com/projects/GuacaMol/56639):
```
wget https://figshare.com/ndownloader/files/13612760 -O guacamol_dataset/guacamol_v1_train.smiles
wget https://figshare.com/ndownloader/files/13612766 -O guacamol_dataset/guacamol_v1_valid.smiles
wget https://figshare.com/ndownloader/files/13612757 -O guacamol_dataset/guacamol_v1_test.smiles
```

And then run the following python script to sample the molecules and calculate the RDKit properties:
```
python prepare_data.py
```