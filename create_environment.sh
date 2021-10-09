#!/bin/zsh
conda create -n $1 -c bioconda -c conda-forge -c pytorch -c stellargraph python=3.7 stellargraph ipykernel snakemake snakefmt flake8 pyflakes pep8 pylint jedi pre-commit gensim scipy seaborn pandas scikit-learn tqdm joblib networkx tensorflow
faiss-gpu
pre-commit install
