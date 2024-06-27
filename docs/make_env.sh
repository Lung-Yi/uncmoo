#!/bin/bash
conda create --name uncmoo python=3.8 -y
conda activate uncmoo

# should be installed already
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch -y     # install using conda to avoid conflict with libxtb
pip install rdkit
conda install -c conda-forge openbabel -y

conda install -c conda-forge xtb-python -y
conda install -c conda-forge crest -y
export XTBHOME=$CONDA_PREFIX
source $CONDA_PREFIX/share/xtb/config_env.bash

pip install --upgrade pip
pip install numpy
pip install pyscf morfeus-ml

# additional packages for polanyi
pip install h5py scikit-learn geometric pyberny loguru wurlitzer sqlalchemy
pip install -i https://test.pypi.org/simple/ geodesic-interpolate
pip install git+https://github.com/kjelljorner/polanyi
pip install chemprop
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
pip install typed-argument-parser # (https://github.com/swansonk14/typed-argument-parser)
pip install janus-ga
