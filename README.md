[![Unit Test & Deploy](https://github.com/skojaku/residual2vec/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/residual2vec/actions/workflows/main.yml)
# Residual2Vec: Debiasing graph embedding using random graphs

This repository contains the code for

- S. Kojaku, J. Yoon, I. Constantino, and Y.-Y. Ahn, Residual2Vec: Debiasing graph embedding using random graphs. NerurIPS (2021). [link will be added when available]

- Preprint (arXiv): https://arxiv.org/abs/2110.07654

- BibTex entry:
```latex
@inproceedings{kojaku2021neurips,
 title={Residual2Vec: Debiasing graph embedding using random graphs},
 author={Sadamori Kojaku and Jisung Yoon and Isabel Constantino and Yong-Yeol Ahn},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {},
 pages = {},
 publisher = {Curran Associates, Inc.},
 volume = {},
 year = {2021}
}
```


# Installation and Usage of `residual2vec` package

```bash
pip install residual2vec
```
The code and instruction for `residual2vec` sits in [libs/residual2vec](libs/residual2vec). See [here](libs/residual2vec/README.md).


# Reproducing the results

We set up Snakemake workflow to reproduce our results. To this end, install [snakemake](https://snakemake.readthedocs.io/en/stable/) and run

```
snakemake --cores <# of cores available> all
```

which will produce all figures for the link prediction and community detection benchmarks. The results for the case study are not generated due to the limitation by our data sharing aggreements.
