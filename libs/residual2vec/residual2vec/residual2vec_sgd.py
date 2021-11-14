"""A python implementation of residual2vec based on the stochastic gradient
descent algorithm. Suitable for large networks.

Usage:

```python
import residual2vec as rv

# Node sampler for the noise distribution for negative sampling
noise_sampler = rv.ConfigModelNodeSampler()

model = rv.residual2vec_sgd(noise_sampler = noise_sampler, window_length = 10)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = model.fit(G).transform(dim = 64)
```

If want to remove the structural bias associated with node labels (i.e., gender):
```python
import residual2vec as rv

group_membership = [0,0,0,0,1,1,1,1] # an array of group memberships of nodes.

# SBMNodeSampler reflects the group membership in sampling
noise_sampler = SBMNodeSampler(window_length = 10, group_membership = group_membership)

model = rv.residual2vec_matrix_factorization(noise_sampler, window_length = 10)
model.fit(G)
emb = model.transform(dim = 64)
```

You can customize the noise_sampler by implementing the following class:

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
"""
import numpy as np
from numba import njit
from scipy import sparse
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from residual2vec import utils
from residual2vec.random_walk_sampler import RandomWalkSampler
from residual2vec.word2vec import NegativeSampling, Word2Vec


class residual2vec_sgd:
    """Residual2Vec based on the stochastic gradient descent.

    .. highlight:: python
    .. code-block:: python
        >>> from residual2vec.residual2vec_sgd import residual2vec_sgd
        >>> net = nx.karate_club_graph()
        >>> model = r2v.Residual2Vec(null_model="configuration", window_length=5, restart_prob=0, residual_type="individual")
        >>> model.fit(net)
        >>> in_vec = model.transform(net, dim = 30)
        >>> out_vec = model.transform(net, dim = 30, return_out_vector=True)
    """

    def __init__(
        self,
        noise_sampler,
        window_length=10,
        batch_size=8,
        num_walks=10,
        walk_length=80,
        p=1,
        q=1,
        cuda=False,
    ):
        """Residual2Vec based on the stochastic gradient descent.

        :param noise_sampler: Noise sampler
        :type noise_sampler: NodeSampler
        :param window_length: length of the context window, defaults to 10
        :type window_length: int
        :param batch_size: Number of batches for the SGD, defaults to 4
        :type batch_size: int
        :param num_walks: Number of random walkers per node, defaults to 100
        :type num_walks: int
        :param walk_length: length per walk, defaults to 80
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        """
        self.window_length = window_length
        self.sampler = noise_sampler
        self.cuda = cuda
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.batch_size = batch_size

    def fit(self, adjmat):
        """Learn the graph structure to generate the node embeddings.

        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: numpy.ndarray or scipy sparse matrix format (csr).
        :return: self
        :rtype: self
        """

        # Convert to scipy.sparse.csr_matrix format
        adjmat = utils.to_adjacency_matrix(adjmat)

        # Set up the graph object for efficient sampling
        self.adjmat = adjmat
        self.n_nodes = adjmat.shape[0]
        self.sampler.fit(adjmat)
        return self

    def transform(self, dim):
        """Generate embedding vectors.

        :param dim: Dimension
        :type dim: int
        :return: Embedding vectors
        :rtype: numpy.ndarray of shape (num_nodes, dim), where num_nodes is the number of nodes.
          Each ith row in the array corresponds to the embedding of the ith node.
        """

        # Set up the embedding model
        PADDING_IDX = self.n_nodes
        model = Word2Vec(
            vocab_size=self.n_nodes + 1, embedding_size=dim, padding_idx=PADDING_IDX
        )
        neg_sampling = NegativeSampling(embedding=model)
        if self.cuda:
            model = model.cuda()

        # Set up the Training dataset
        dataset = TripletDataset(
            adjmat=self.adjmat,
            num_walks=self.num_walks,
            window_length=self.window_length,
            noise_sampler=self.sampler,
            padding_id=PADDING_IDX,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Training
        optim = Adam(model.parameters())
        pbar = tqdm(dataloader)
        for iword, owords, nwords in pbar:
            loss = neg_sampling(iword, owords, nwords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        self.in_vec = model.ivectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        self.out_vec = model.ovectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        return self.in_vec


class TripletDataset(Dataset):
    """Dataset for training word2vec with negative sampling."""

    def __init__(
        self,
        adjmat,
        num_walks,
        window_length,
        noise_sampler,
        padding_id,
        walk_length=40,
        p=1.0,
        q=1.0,
    ):
        """Dataset for training word2vec with negative sampling.

        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: scipy sparse matrix format (csr).
        :param num_walks: Number of random walkers per node
        :type num_walks: int
        :param window_length: length of the context window
        :type window_length: int
        :param noise_sampler: Noise sampler
        :type noise_sampler: NodeSampler
        :param padding_id: Index of the padding node
        :type padding_id: int
        :param walk_length: length per walk, defaults to 40
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        """
        self.adjmat = adjmat
        self.num_walks = num_walks
        self.window_length = window_length
        self.noise_sampler = noise_sampler
        self.walk_length = walk_length
        self.padding_id = padding_id
        self.rw_sampler = RandomWalkSampler(adjmat, walk_length=walk_length, p=p, q=q)
        self.node_order = np.random.choice(
            adjmat.shape[0], adjmat.shape[0], replace=False
        )
        self.n_nodes = adjmat.shape[0]

        # Counter and Memory
        self.rw_id = 0
        self.t_walk = 0  #
        self.walk = None

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.n_nodes * self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.t_walk == len(self.walk):
            self._generate_samples()

        center = self.walk[self.t_walk]
        context = _get_context(
            self.walk,
            len(self.walk),
            self.t_walk,
            self.window_length,
            padding_id=self.padding_id,
        )
        random_context = self.noise_sampler.sampling(center, len(context))
        self.t_walk += 1
        return center, context.astype(np.int64), random_context.astype(np.int64)

    def _generate_samples(self):
        walk = self.rw_sampler.sampling(self.node_order[self.rw_id])
        self.rw_id = (self.rw_id + 1) % self.n_nodes
        self.walk = walk
        self.t_walk = 0


@njit(nogil=True)
def _get_context(walk, walk_len, t_walk, window_length, padding_id):
    retval = padding_id * np.ones(2 * window_length, dtype=np.int64)
    for i in range(window_length):
        if t_walk - 1 - i < 0:
            break
        retval[window_length - 1 - i] = walk[t_walk - 1 - i]

    for i in range(window_length):
        if t_walk + 1 + i >= walk_len:
            break
        retval[window_length + i] = walk[t_walk + 1 + i]
    return retval
