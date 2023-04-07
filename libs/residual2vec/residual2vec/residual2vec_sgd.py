# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-04-29 21:31:09
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-07 12:28:22
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

    def sampling(self, center_node=None, context_node = None, size = None):
        # Sample context nodes from the graph for center nodes
        #:param center_node: ID of center node
        #:type center_node: int
        #:param size: number of samples per center node
        #:type size: int
        pass

The sampling function should return the random center nodes and random context nodes.
The size should be the same as center_node if provided, otherwise should follow the specified size by `size`.
If `center_node` is provided, the returned center_node should be the same. The function is expected to produce random context nodes conditioned on the center nodes. If not provided, however, the center nodes need to be sampled from the given network
"""
import random

import numpy as np
import torch
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
        batch_size=256,
        num_walks=5,
        walk_length=40,
        p=1,
        q=1,
        device="cpu",
        buffer_size=100000,
        context_window_type="double",
        miniters=200,
        learn_joint_probability=False,
        unconnected_negatives=False,
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
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        :param context_window_type: The type of context window. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" and ="right" specifies that extends left and right, respectively.
        :type context_window_type: str, optional
        :param miniter: Minimum number of iterations, defaults to 200
        :type miniter: int, optional
        :param learn_joint_probability: Set `learn_joint_probability=True` to learn the joint probability P(i,j) of random walks, instead of transition probability, P(j | i). Default to False.
        :type learn_joint_probability: bool, optional
        :param unconnected_negatives: If set True, we sample the unconnected node pairs.
        :type unconnected_negatives: bool, optional
        """
        self.window_length = window_length
        self.sampler = noise_sampler
        self.device = device
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.miniters = miniters
        self.context_window_type = context_window_type
        self.learn_joint_probability = learn_joint_probability
        self.unconnected_negatives = unconnected_negatives

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
            vocab_size=self.n_nodes + 1,
            embedding_size=dim,
            padding_idx=PADDING_IDX,
            device=self.device,
        )
        neg_sampling = NegativeSampling(embedding=model)
        if self.device != "cpu":
            model = model.to(self.device)

        # Set up the Training dataset
        adjusted_num_walks = np.ceil(
            self.num_walks
            * np.maximum(
                1,
                self.batch_size
                * self.miniters
                / (self.n_nodes * self.num_walks * self.walk_length),
            )
        ).astype(int)
        dataset = TripletDataset(
            adjmat=self.adjmat,
            num_walks=adjusted_num_walks,
            window_length=self.window_length,
            noise_sampler=self.sampler,
            padding_id=PADDING_IDX,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
            buffer_size=self.buffer_size,
            context_window_type=self.context_window_type,
            learn_joint_probability=self.learn_joint_probability,
            unconnected_negatives=self.unconnected_negatives,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        # Training
        optim = Adam(model.parameters(), lr=0.003)
        # scaler = torch.cuda.amp.GradScaler()
        pbar = tqdm(dataloader, miniters=100)
        for batch in pbar:
            # optim.zero_grad()
            for param in model.parameters():
                param.grad = None

            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)
            # with torch.cuda.amp.autocast():

            loss = neg_sampling(*batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
        context_window_type="double",
        buffer_size=100000,
        learn_joint_probability=True,
        unconnected_negatives=False,
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
        :param context_window_type: The type of context window. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" and ="right" specifies that extends left and right, respectively.
        :type context_window_type: str, optional
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        :param learn_joint_probability: If set True, the dataset will additionally generate the random center nodes as return it at the fourth variable.
        :type learn_joint_probability: bool, optional
        :param unconnected_negatives: If set True, we sample the unconnected node pairs.
        :type unconnected_negatives: bool, optional
        """
        self.adjmat = adjmat
        self.num_walks = num_walks
        self.window_length = window_length
        self.noise_sampler = noise_sampler
        self.walk_length = walk_length
        self.padding_id = padding_id
        self.learn_joint_probability = learn_joint_probability
        self.unconnected_negatives = unconnected_negatives
        self.context_window_type = {"double": 0, "left": -1, "right": 1}[
            context_window_type
        ]
        self.rw_sampler = RandomWalkSampler(
            adjmat, walk_length=walk_length, p=p, q=q, padding_id=padding_id
        )
        self.node_order = np.random.choice(
            adjmat.shape[0], adjmat.shape[0], replace=False
        )
        self.n_nodes = adjmat.shape[0]

        self.ave_deg = adjmat.sum() / adjmat.shape[0]

        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_centers = None
        self.random_contexts = None

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.n_nodes * self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id]
        cont = self.contexts[self.sample_id].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id].astype(np.int64)

        if self.learn_joint_probability:
            rand_center = self.random_centers[self.sample_id].astype(np.int64)
            self.sample_id += 1

            return center, cont, rand_cont, rand_center
        else:
            self.sample_id += 1
            return center, cont, rand_cont

    def _generate_samples(self):
        next_scanned_node_id = np.minimum(
            self.scanned_node_id + self.buffer_size, self.n_nodes
        )
        walks = self.rw_sampler.sampling(
            self.node_order[self.scanned_node_id : next_scanned_node_id]
        )
        self.centers, self.contexts = _get_center_context(
            context_window_type=self.context_window_type,
            walks=walks,
            n_walks=walks.shape[0],
            walk_len=walks.shape[1],
            window_length=self.window_length,
            padding_id=self.padding_id,
        )
        self.random_centers, self.random_contexts = self.noise_sampler.sampling(
            center_nodes=None if self.learn_joint_probability else self.centers,
            size=len(self.centers),
        )

        if self.unconnected_negatives:
            s = (
                np.array(
                    self.adjmat[(self.random_centers, self.random_contexts)]
                ).reshape(-1)
                == 0
            )
            self.random_centers, self.random_contexts = (
                self.random_centers[s],
                self.random_contexts[s],
            )
            self.centers, self.contexts = self.centers[s], self.contexts[s]

        self.n_sampled = len(self.centers)
        self.scanned_node_id = next_scanned_node_id % self.n_nodes
        self.sample_id = 0


def _get_center_context(
    context_window_type, walks, n_walks, walk_len, window_length, padding_id
):
    """Get center and context pairs from a sequence
    window_type = {-1,0,1} specifies the type of context window.
    window_type = 0 specifies a context window of length window_length that extends both
    left and right of a center word. window_type = -1 and 1 specifies a context window
    that extends either left or right of a center word, respectively.
    """
    if context_window_type == 0:
        center, context = _get_center_double_context_windows(
            walks, n_walks, walk_len, window_length, padding_id
        )
    elif context_window_type == -1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
        )
    elif context_window_type == 1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=False
        )
    else:
        raise ValueError("Unknown window type")
    center = np.outer(center, np.ones(context.shape[1]))
    center, context = center.reshape(-1), context.reshape(-1)
    s = (center != padding_id) * (context != padding_id)
    center, context = center[s], context[s]
    order = np.arange(len(center))
    random.shuffle(order)
    return center[order].astype(int), context[order].astype(int)


@njit(nogil=True)
def _get_center_double_context_windows(
    walks, n_walks, walk_len, window_length, padding_id
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones(
        (n_walks * walk_len, 2 * window_length), dtype=np.int64
    )
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        for i in range(window_length):
            if t_walk - 1 - i < 0:
                break
            contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]

        for i in range(window_length):
            if t_walk + 1 + i >= walk_len:
                break
            contexts[start:end, window_length + i] = walks[:, t_walk + 1 + i]

    return centers, contexts


@njit(nogil=True)
def _get_center_single_context_window(
    walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        if is_left_window:
            for i in range(window_length):
                if t_walk - 1 - i < 0:
                    break
                contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]
        else:
            for i in range(window_length):
                if t_walk + 1 + i >= walk_len:
                    break
                contexts[start:end, i] = walks[:, t_walk + 1 + i]
    return centers, contexts
