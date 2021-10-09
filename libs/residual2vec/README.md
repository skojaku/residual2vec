[![Unit Test & Deploy](https://github.com/skojaku/residual2vec/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/residual2vec/actions/workflows/main.yml)

# Python package for residual2vec graph embedding algorithm

residual2vec is an algorithm to embed networks to a vector space while controlling for various structural properties such as degree. If you use this package, please cite:

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

## Install

```bash
pip install residual2vec
```

### Requirements

This code is tested in Python 3.7 and 3.8, and has dependencies with
the following packages:

```
- numpy==1.20.3
- scipy==1.7.1
- scikit-learn==1.0
- faiss-cpu==1.7.0
```


## Example

```python
import residual2vec as rv

model = rv.residual2vec(window_length = 10, group_membership = None)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = model.fit(G).transform(dim = 64)
```
- `G`: adjacency matrix of the input graph. [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) or [scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) can be accepted.
- `window_length`: the length of context window.
- `group_membership`: an array of node labels. Used to debias the structural bias correlated with the node labels.
- `dim`: Dimension of the embedding
- `emb`: 2D numpy array of shape (`N`, `dim`), where `N` is the number of nodes. The `i`th row in the array (i.e., `emb[i, :]`) represents the embedding vector of the `i`th node in the given adjacency matrix `G`.
