# Residual2Vec: Debiasing graph embedding using random graphs

This repository contains the code for

- S. Kojaku, J. Yoon, I. Constantino, and Y.-Y. Ahn, Residual2Vec: Debiasing graph embedding using random graphs. NerurIPS (2021). [link will be added when available]

- Preprint (arXiv): [link to arXiv]

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


## Installation and Usage of `residual2vec` package

The code for `residual2vec` sits in [libs/residual2vec](libs/residual2vec). See the [README](libs/residual2vec/README.md) in the diretory for details.


## Reproducing the results

We set up Snakemake workflow to reproduce our results. To this end, install [snakemake](https://snakemake.readthedocs.io/en/stable/) and run

```
snakemake --cores <# of cores available> all
```

which will produce all figures for the link prediction and community detection benchmarks. The results for the case study are not generated due to the limitation by our data sharing aggreements.
