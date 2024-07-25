# Uncertainty-Aware Machine Learning Approaches for Robust Molecular Design
This is a uncertainty-aware molecular design framework benchmarked on the Tartarus platform (https://github.com/aspuru-guzik-group/Tartarus).

The manuscript of this repository is in preparation.

<!-- ![alt text](docs/ms_1_overview_a.svg) -->
<img src="docs/ms_1_overview_a.svg" style="width: 100%; height: auto;">

## OS Requirements
This repository requires to operate on **Linux** operating system.

## Python Dependencies
* Python (version >= 3.8)
* rdkit (version >= 2022.3.4)
* torch (versioin >= 1.12.1)
* matplotlib (version >=3.3.4)
* numpy (version >= 1.16.4)

## Installation
```
bash docs/make_env.sh
```

## Design benchmarks cutoff values for single- and multi-objective tasks
| Design Benchmark | Objective | Optimal Value | Top-15% Cutoff for Multi-objective Task |
|------------------|-----------|---------------|----------------------------------------|
| **Organic Emitters** | | | |
| | Singlet-triplet gap (↓) | 0.00249 (eV) | 0.571 |
| | Oscillator strength (↑) | 2.97 (-) | 0.171 |
| | Absolute difference of vertical excitation energy (↓) | 0.0573 (eV) | 1.62 |
| **Protein Ligands** | | | |
| | 1SYH score (↓) | -10.0 (-) | - |
| | 6Y2F score (↓) | -10.0 (-) | - |
| | 4LDE score (↓) | -9.8 (-) | - |
| **Reaction Substrates** | | | |
| | Activation energy (↓) | 64.9 (kcal/mol) | 87.0 (maximization for multi-objective) |
| | Reaction energy (↓) | -34.4 (kcal/mol) | -5.46 |

## Download the results in the paper for analysis (optional)
```
gdown --id '1tahu9C0EwlvCNOIXjvz96N2e34TGawyV' --output RESULTS.tar.gz
tar zxvf RESULTS.tar.gz
```

## Single-objective molecular optimization example (probability improvement)
```
DATASET=docking
OBJECTIVE="4lde"
TARGET_NAME="4lde score"
CUTOFF=-10.001
DATA_PATH="docking_normalized.csv"
METHOD=uncertainty
N_SAMPLE=10000
FOLD=1

python janus_benchmark.py --benchmark_data $DATASET \
                          --fitness_method $METHOD --n_sample $N_SAMPLE \
                          --result_path RESULTS/${DATASET}_${OBJECTIVE}/janus_${METHOD}_${FOLD} \
                          --start_smiles_path Tartarus/datasets/$DATA_PATH \
                          --surrogate_model_path chemprop_unc/save_models/$DATASET/evidential/fold_0/ \
                          --target_columns $TARGET_NAME \
                          --target_cutoff CUTOFF \
                          --batch_pred \
                          --target_objective minimize | tee log_${METHOD}_${DATASET}_${OBJECTIVE}_${FOLD}.txt
```
The single-objective molecule design results can then be found in the:
`RESULTS/docking_4lde/janus_uncertainty_1/population_explore.txt`
and
`RESULTS/docking_4lde/janus_uncertainty_1/fitness_explore.txt`

Further analysis of all the single-objective optimization results refers to:

https://github.com/Lung-Yi/uncmoo/blob/main/scripts_analyze/analyze_single_objective.ipynb


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
The multi-objective molecule design results can then be found in the:
`RESULTS/organic_emitter/janus_uncertainty_1/population_explore.txt`
and
`RESULTS/organic_emitter/janus_uncertainty_1/fitness_explore.txt`

Further analysis of all the multi-objective optimization results refers to:

https://github.com/Lung-Yi/uncmoo/blob/main/scripts_analyze/plot_multi_objective.ipynb

## Objective function selection for calculating the different fitness
### Single-objective
| Argument for `--fitness_method` | Method                         |
|---------------------------------|--------------------------------|
| `uncertainty`                   | Probabilistic Improvement Optimization (PIO) |
| `scaler`                        | Direct Objective Maximization (DOM)          |
| `expected_improvement`          | Expected Improvement (EI)                    |

### Multi-objective
| Argument for `--fitness_method` | Method               |
|--------------|--------------------------------|
| `uncertainty`| Probabilistic Improvement Optimization (PIO)  |
| `scaler`     | Weighted Sum (WS)                             |
| `utopian`    | Normalized Manhattan Distance (NMD)           |
| `hybrid`     | Hybrid Approach (NMD-WS)                      |
