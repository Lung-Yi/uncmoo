# Uncertainty-Aware Machine Learning Approaches for Robust Molecular Design
This is a uncertainty-aware molecular design framework benchmarked on the Tartarus platform (https://github.com/aspuru-guzik-group/Tartarus).

The manuscript of this repository is in preparation.

![alt text](docs/ms_1_overview.svg)

## Installation
```
bash docs/make_env.sh
```

## Download the results in the paper for analysis (optional)
```
gdown --id '1tahu9C0EwlvCNOIXjvz96N2e34TGawyV' --output RESULTS.tar.gz
tar zxvf RESULTS.tar.gz
```

## Single-objective molecular optimization example (probability improvement)
```
DATASET=docking
DATA_PATH="docking_normalized.csv"
METHOD=uncertainty
N_SAMPLE=10000
FOLD=1

python janus_benchmark.py --benchmark_data $DATASET \
                          --fitness_method $METHOD --n_sample $N_SAMPLE \
                          --result_path RESULTS/${DATASET}_4lde/janus_${METHOD}_${FOLD} \
                          --start_smiles_path Tartarus/datasets/$DATA_PATH \
                          --surrogate_model_path chemprop_unc/save_models/$DATASET/evidential/fold_0/ \
                          --target_columns "4lde score" \
                          --target_cutoff -10.001 \
                          --batch_pred \
                          --target_objective minimize | tee log_${METHOD}_${DATASET}_4lde_${FOLD}.txt
```

## Multi-objective molecular optimization example (probability improvement)
```
PARENT="chemprop_unc/save_models"
DATASET=organic_emitter
DATA_PATH="gdb13_normalized.csv"
ALPHATBET="gdb13_alphabet.txt"
SUFFIX="fold_0"
METHOD=uncertainty
N_SAMPLE=10000

FOLD=1
python janus_benchmark.py --benchmark_data $DATASET \
                          --fitness_method $METHOD --n_sample $N_SAMPLE \
                          --result_path RESULTS/$DATASET/janus_${METHOD}_${FOLD} \
                          --start_smiles_path Tartarus/datasets/$DATA_PATH \
                          --surrogate_model_path $PARENT/$DATASET/ensemble_mve/$SUFFIX \
                          --target_columns "singlet-triplet value" "oscillator strength" "abs_diff_vee" \
                          --target_cutoff 0.57068 0.1706886 1.615238 \
                          --batch_pred \
                          --alphabet_path Tartarus/datasets/$ALPHATBET \
                          --target_objective minimize maximize minimize | tee log_${METHOD}_${DATASET}_${FOLD}.txt
```
