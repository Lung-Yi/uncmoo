#!/bin/bash
#PBS -l select=1:ncpus=1
#PBS -q workq

source ~/.bashrc
conda activate tartarus
cd /home/lungyi/uncmoo/scripts


DATASET=$$$DATASET$$$
SMILES='$$$SMILES$$$'

python single_score_calc.py --dataset $DATASET --smiles $SMILES


conda deactivate