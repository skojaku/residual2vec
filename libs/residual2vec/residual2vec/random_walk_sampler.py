"""Graph module to store a network and generate random walks from it."""
from collections.abc import Iterable

import numpy as np
from numba import njit

from residual2vec import utils


class RandomWalkSampler:
    """Module for generating a sentence using random walks.

    .. highlight:: python
    .. code-block:: python
        >>> from residual2vec.random_walk_sampler import RandomWalkSampler
        >>> net = nx.adjacency_matrix(nx.karate_club_graph())
        >>> sampler = RandomWalkSampler(net)
        >>> walk = sampler.sampling(start=12)
        >>> print(walk) # [12, 11, 10, 9, ...]
    """

    def __init__(self, adjmat, walk_length=40, p=1, q=1):
        """Random Walk Sampler.

        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: scipy sparse matrix format (csr).
        :param walk_length: length per walk, defaults to 40
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        """
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.weighted = (~np.isclose(np.min(adjmat.data), 1)) or (
            ~np.isclose(np.max(adjmat.data), 1)
        )

        adjmat.sort_indices()
        self.indptr = adjmat.indptr.astype(np.int64)
        self.indices = adjmat.indices.astype(np.int64)
        if self.weighted:
            data = adjmat.data / adjmat.sum(axis=1).A1.repeat(np.diff(self.indptr))
            self.data = utils._csr_row_cumsum(self.indptr, data)

    def sampling(self, start):
        """Sample a random walk path.

        :param start: ID of the starting node
        :type start: int
        :return: array of visiting nodes
        :rtype: np.ndarray
        """
        if self.weighted:
            walk = _random_walk_weighted(
                self.indptr,
                self.indices,
                self.data,
                self.walk_length,
                self.p,
                self.q,
                start
                if isinstance(start, Iterable)
                else np.array([start]).astype(np.int64),
            )
        else:
            walk = _random_walk(
                self.indptr,
                self.indices,
                self.walk_length,
                self.p,
                self.q,
                start
                if isinstance(start, Iterable)
                else np.array([start]).astype(np.int64),
            )
        if isinstance(start, Iterable):
            return walk.astype(np.int64)
        else:
            return walk[0].astype(np.int64)


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, ts):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty((len(ts), walk_length), dtype=indices.dtype)

    for walk_id, t in enumerate(ts):
        walk[walk_id, 0] = t
        walk[walk_id, 1] = np.random.choice(_neighbors(indptr, indices, t))
        for j in range(2, walk_length):
            neighbors = _neighbors(indptr, indices, walk[walk_id, j - 1])
            if p == q == 1:
                # faster version
                walk[walk_id, j] = np.random.choice(neighbors)
                continue
            while True:
                new_node = np.random.choice(neighbors)
                r = np.random.rand()
                if new_node == walk[walk_id, j - 2]:
                    if r < prob_0:
                        break
                elif _isin_sorted(
                    _neighbors(indptr, indices, walk[walk_id, j - 2]), new_node
                ):
                    if r < prob_1:
                        break
                elif r < prob_2:
                    break
            walk[walk_id, j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, ts):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty((len(ts), walk_length), dtype=indices.dtype)

    for walk_id, t in enumerate(ts):
        walk[walk_id, 0] = t
        walk[walk_id, 1] = _neighbors(indptr, indices, t)[
            np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
        ]
        for j in range(2, walk_length):
            neighbors = _neighbors(indptr, indices, walk[walk_id, j - 1])
            neighbors_p = _neighbors(indptr, data, walk[walk_id, j - 1])
            if p == q == 1:
                # faster version
                walk[walk_id, j] = neighbors[
                    np.searchsorted(neighbors_p, np.random.rand())
                ]
                continue
            while True:
                new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
                r = np.random.rand()
                if new_node == walk[walk_id, j - 2]:
                    if r < prob_0:
                        break
                elif _isin_sorted(
                    _neighbors(indptr, indices, walk[walk_id, j - 2]), new_node
                ):
                    if r < prob_1:
                        break
                elif r < prob_2:
                    break
            walk[walk_id, j] = new_node
    return walk
