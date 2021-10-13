#!/bin/zsh
conda create -n $1 -c conda-forge python=3.6 gensim scipy seaborn pandas scikit-learn tqdm joblib networkx
pre-commit install
