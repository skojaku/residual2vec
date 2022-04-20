[![Unit Test & Deploy](https://github.com/skojaku/residual2vec/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/residual2vec/actions/workflows/main.yml)

# Python package for residual2vec graph embedding algorithm

residual2vec is an algorithm to embed networks to a vector space while controlling for various structural properties such as degree. If you use this package, please cite:

- S. Kojaku, J. Yoon, I. Constantino, and Y.-Y. Ahn, Residual2Vec: Debiasing graph embedding using random graphs. NerurIPS (2021). [link will be added when available]

- [Preprint (arXiv)](https://arxiv.org/abs/2110.07654)

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

## Install

```bash
pip install residual2vec
```

### Requirements

This code is tested in Python 3.7 and 3.8, and has dependencies with
the following packages:

```
- numpy==1.19.0
- scipy==1.7.1
- scikit-learn==1.0
- faiss-cpu==1.7.0
- numba==0.50.0
- torch==1.10.0
- tqdm==4.48.2
```


## Example

residual2vec has two versions, one optimized with a matrix factorization, and the other optimized with a stochatic gradient descent aglorithm.

The residual2vec with a matrix factorization is used in the original paper and runs faster than the other version for networks of upto 100k nodes.

```python
import residual2vec as rv

model = rv.residual2vec_matrix_factorization(window_length = 10, group_membership = None)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = model.fit(G).transform(dim = 64)
```
- `G`: adjacency matrix of the input graph. [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) or [scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) can be accepted.
- `window_length`: the length of context window.
- `group_membership`: an array of node labels. Used to debias the structural bias correlated with the node labels.
- `dim`: Dimension of the embedding
- `emb`: 2D numpy array of shape (`N`, `dim`), where `N` is the number of nodes. The `i`th row in the array (i.e., `emb[i, :]`) represents the embedding vector of the `i`th node in the given adjacency matrix `G`.


A limitation of the matrix-factorization-based implementation is that it is memory demanding, especially for dense or large networks.
The other version is implemented to circumvent this problem by using the stochastic gradient descent (SGD) algorithm, that
incrementally updates the embedding with a small chunk of data instead of deriving the whole embedding in one go.

```python
import residual2vec as rv

noise_sampler = rv.ConfigModelNodeSampler() # sampler for the negative sampling

model = rv.residual2vec_sgd(noise_sampler, window_length = 10)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = model.fit(G).transform(dim = 64)
```

The `residual2vec_sgd` has an additional argument `noise_sampler`, which is a class that samples context nodes for a given center node.
Several samplers are implemented in this package:
- `ErdosRenyiNodeSampler`: Sampler based on the Erdos Renyi random graph (i.e., sample context node with the same probability)
- `ConfigModelNodeSampler`: Sampler based on the configuration model (i.e., sample context node with probability proportional to its degree)
- `SBMNodeSampler`: Sampler based on the stochastic block model (i.e., sample context node using the stochastic block model)
- `ConditionalContextSampler`: Sampling a random context node conditioned on the group to which a given context node blongs. The group membership needs to be given when creating this instance (experimental).

The `SBMNodeSampler` is useful to negate the bias due to a group structure in networks (i.e., structure correlated with a discrete label of nodes):

```python
import residual2vec as rv

group_membership = [0,0,0,0,1,1,1,1]
noise_sampler = rv.SBMNodeSampler(window_length = 10, group_membership=group_membership) # sampler for the negative sampling

model = rv.residual2vec_sgd(noise_sampler, window_length = 10)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = model.fit(G).transform(dim = 64)
```

An added bonus for the SGD-based approach is that it offers a way to customize the noise distribution, which is useful to debias a particular bias in embedding.
Implement the following class inherited from `rv.NodeSampler`:

```python
import residual2vec as rv
class CustomNodeSampler(rv.NodeSampler):
    def fit(self, A):
        #Fit the sampler
        #:param A: adjacency matrix
        #:type A: scipy.csr_matrix
        pass

    def sampling(self, center_node, n_samples):
        #Sample context nodes from the graph for center nodes
        #:param center_node: ID of center node
        #:type center_node: int
        #:param n_samples: number of samples per center node
        #:type n_samples: int
        pass
```

See the `residual2vec/node_samplers` for examples.
