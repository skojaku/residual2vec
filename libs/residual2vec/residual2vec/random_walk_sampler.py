"""Graph module to store a network and generate random walks from it."""
import numpy as np
from numba import njit

from residual2vec import utils


class RandomWalkSampler:
    def __init__(self, A, walk_length=40, p=1, q=1):
        self.walk_length = walk_length
        self.p = p
        self.q = q

        self.weighted = (~np.isclose(np.min(A.data), 1)) or (
            ~np.isclose(np.max(A.data), 1)
        )

        A.sort_indices()
        self.indptr = A.indptr
        self.indices = A.indices
        if self.weighted:
            data = A.data / A.sum(axis=1).A1.repeat(np.diff(self.indptr))
            self.data = utils._csr_row_cumsum(self.indptr, data)

    def sampling(self, start):
        if self.weighted:
            walk = _random_walk_weighted(
                self.indptr,
                self.indices,
                self.data,
                self.walk_length,
                self.p,
                self.q,
                start,
            )
        else:
            walk = _random_walk(
                self.indptr, self.indices, self.walk_length, self.p, self.q, start
            )
        return walk.astype(np.int64)


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    walk[1] = np.random.choice(_neighbors(indptr, indices, t))
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = np.random.choice(neighbors)
            continue
        while True:
            new_node = np.random.choice(neighbors)
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    walk[1] = _neighbors(indptr, indices, t)[
        np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    ]
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        neighbors_p = _neighbors(indptr, data, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            continue
        while True:
            new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk
