"""A python implementation of residual2vec based on the block approximation.

Usage:

```python
import residual2vec as rv

model = rv.residual2vec(window_length = 10, group_membership = None)
model.fit(G)
emb = model.transform(dim = 64)
# or equivalently emb = mode.fit(G).transform(dim = 64)
```

If want to remove the structural bias associated with node labels (i.e., gender):
```python
import residual2vec as rv

group_membership = [0,0,0,0,1,1,1,1] # an array of group memberships of nodes.
model = rv.residual2vec(window_length = 10, group_membership = group_membership)
model.fit(G)
emb = model.transform(dim = 64)
```
"""

import faiss
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from residual2vec import utils


class residual2vec:
    def __init__(
        self, window_length=10, group_membership=None, num_blocks=1000,
    ):
        """Residual2Vec.

        :param window_length: window length, defaults to 10
        :type window_length: int, optional
        :param group_membership:  Group membership of nodes.
            The order of nodes should be the same order with G.nodes() if the given network is in networkx.Graph object format.
            If the given network is in scipy.sparse matrix format, the order should be in the ascending order of node index., defaults to None
        :type group_membership: [type], optional
        :param num_blocks: Number of blocks for block approximation, defaults to 1000
        :type num_blocks: int, optional.

        .. highlight:: python
        .. code-block:: python
            >>> import residual_node2vec as r2v
            >>> net = nx.karate_club_graph()
            >>> model = r2v.Residual2Vec(null_model="configuration", window_length=5, restart_prob=0, residual_type="individual")
            >>> model.fit(net)
            >>> in_vec = model.transform(net, dim = 30)
            >>> out_vec = model.transform(net, dim = 30, return_out_vector=True)
        """

        if group_membership is not None:
            num_blocks = np.maximum(num_blocks, len(set(group_membership)))

        self.window_length = window_length
        self.num_blocks = num_blocks

        if group_membership is None:
            self.group_membership = None
        else:
            self.group_membership = np.unique(group_membership, return_inverse=True)[
                1
            ]  # reindex

        self.U = None  # Residual In-vector
        self.alpha = 0.5  # Loading of the singular values.

    def fit(self, adjmat):
        """Learn the graph structure to generate the node embeddings.

        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: numpy.ndarray or scipy sparse matrix format (csr).
        :return: self
        :rtype: self
        """

        # Convert to scipy.sparse.csr_matrix format
        A = utils.to_adjacency_matrix(adjmat)

        # Block approximation
        if self.window_length > 1:
            ghat = _find_blocks_by_sbm(A, self.num_blocks)
        else:
            ghat = np.arange(A.shape[0])

        if self.group_membership is None:
            self.group_membership = np.zeros(A.shape[0], dtype=int)

        # Construct the truncated residual2vec
        self.truncated_R = _truncated_residual_matrix(
            A,
            group_ids=ghat,
            group_ids_null=self.group_membership,
            window_length=self.window_length,
        )

        return self

    def transform(self, dim):
        """Generate embedding vectors.

        :param dim: Dimension
        :type dim: int
        :return: Embedding vectors
        :rtype: numpy.ndarray of shape (num_nodes, dim), where num_nodes is the number of nodes.
          Each ith row in the array corresponds to the embedding of the ith node.
        """
        svd = TruncatedSVD(n_components=dim, algorithm="randomized")
        svd.fit(self.truncated_R.T)
        U, lam = svd.components_, svd.singular_values_
        U = U.T @ np.diag(np.power(lam, self.alpha))

        order = np.argsort(lam)[::-1]
        self.U = U[:, order]

        return U


# =================
# Private functions
# =================
def _find_blocks_by_sbm(A, K, directed=False):
    """Fast community detection by SCORE (see the reference below).

    :param A: scipy sparse matrix
    :type A: sparse.csr_matrix
    :param K: number of communities
    :type K: int
    :param directed: whether to cluster directed or undirected, defaults to False
    :type directed: bool, optional
    :return: group membership
    :rtype: numpy.ndarray

    Reference:
        Jiashun Jin. "Fast community detection by SCORE."
        Ann. Statist. 43 (1) 57 - 89, February 2015.
        https://doi.org/10.1214/14-AOS1265
    """

    # Return every node as a community if the number of communities
    # exceeds the number of nodes
    if K >= (A.shape[0] - 1):
        cids = np.arange(A.shape[0])
        return cids

    # Embed the nodes using the eigen-decomposition of the adjacency matrix.
    u, s, v = utils.rSVD(A, dim=K)
    if directed:
        u = np.hstack([u, v])

    # Normlize to have unit norm for spherical clustering
    u = np.einsum("ij,i->ij", u, 1 / np.maximum(1e-12, np.linalg.norm(u, axis=1)))

    if (u.shape[0] / K) < 10:
        niter = 1
    else:
        niter = 10

    km = faiss.Kmeans(u.shape[1], K, niter=niter, spherical=True)
    km.train(u)
    _, cids = km.index.search(u, 1)
    cids = np.array(cids).reshape(-1)

    return np.array(cids).reshape(-1)


def _truncated_residual_matrix(A, group_ids, group_ids_null, window_length):
    """Compute the truncated residual matrix using the block approximation.

    :param A: Adjacency matrix
    :type A: scipy.csr_matrix
    :param group_ids: group membership
    :type group_ids: np.ndarray
    :param group_ids_null: group membership for random graphs
    :type group_ids_null: np.ndarray
    :param window_length: window length
    :type window_length: int
    :return: Truncated residual matrix
    :rtype: sparse.csr_matrix
    """

    # ----------------------------------
    # Prep. variables to compute S and L
    # ----------------------------------

    # Transition matrix
    P = utils.row_normalize(A)

    # Calculating the parameters for dcSBM for the block approximation
    #   Din: Din[k] = the number of edges pointing to nodes in the kth group
    #   Psbm: Block-wise transition matrix
    #   node2g: mapping matrix. node2g[i,k] = 1 if node i belongs to the kth group. Otherwise 0.
    Din, Psbm, node2g = _get_sbm_params(A, group_ids)

    # Calculating the parameters for dcSBM for random graphs
    Din_null, Psbm_null, node2gnull = _get_sbm_params(A, group_ids_null)

    # Prep. mapping matrices
    #   g2pair: group_ids <-> group_pair_ids
    #   gnull2pair: group_ids_null <-> group_pair_ids
    #   node2pair: node <-> group_pair_ids
    group_pair_ids = np.unique(  # pair (group_ids, group_ids_null) and reindex
        utils.pairing(group_ids, group_ids_null), return_inverse=True
    )[1]
    g2pair = utils.to_member_matrix(group_pair_ids, node_ids=group_ids)
    gnull2pair = utils.to_member_matrix(group_pair_ids, node_ids=group_ids_null)
    node2pair = utils.to_member_matrix(group_pair_ids)

    # ----------------------------------
    # Calculate the elements of S
    # ----------------------------------

    # Compute the sum power \sum_{t=1} ^{T-1} Psbm^t
    Psbm_pow = utils.matrix_sum_power(Psbm, window_length - 1)
    Psbm_null_pow = utils.matrix_sum_power(Psbm_null, window_length)

    P_node2g_Psbm_pow = P @ node2g @ Psbm_pow
    node2gnull_Psbm_null_pow = node2gnull @ Psbm_null_pow
    p_rows, p_cols, prc = sparse.find(P)
    src = utils.safe_log(
        prc + P_node2g_Psbm_pow[p_rows, group_ids[p_cols]]
    ) - utils.safe_log(P_node2g_Psbm_pow[p_rows, group_ids[p_cols]])

    # ----------------------------------
    # Calculate matrix L
    # ----------------------------------

    h1 = (
        utils.safe_log(
            sparse.csr_matrix(P_node2g_Psbm_pow)
            @ sparse.diags(1 / np.maximum(1e-32, Din))
        )
        @ g2pair
    )
    h2 = (
        utils.safe_log(
            sparse.csr_matrix(node2gnull_Psbm_null_pow)
            @ sparse.diags(1 / np.maximum(1e-32, Din_null))
        )
        @ gnull2pair
    )
    h = h1 - h2

    # Truncation
    h_truncated = h.copy()
    h_truncated.data[h_truncated.data < 0] = 0  # truncation at h
    h_truncated.eliminate_zeros()

    # Construct L
    L_truncated = h_truncated @ sparse.csr_matrix(node2pair.T)

    # ---------------------------------------
    # Calculate truncated residual matrix, R
    # ---------------------------------------

    R_truncated = L_truncated.copy()
    R_truncated[(p_rows, p_cols)] = h[(p_rows, group_pair_ids[p_cols])] + src
    return R_truncated


def _get_sbm_params(A, group_ids):
    """Calculate the ML estimate of the parameters for the dcSBM.

    :param A: adjacency matrix
    :type A: scipy.csr_matrix
    :param group_ids: group membership
    :type group_ids: np.ndarray
    :return: Din: Degree of groups. Psbm: group-level transition matrix. U: membership matrix.
    :rtype: Din: nd.array. Psbm: np.ndarray. U: sparse.csr_matrix
    """
    U = utils.to_member_matrix(group_ids)
    Lambda = utils.row_normalize((U.T @ A @ U).toarray())
    Din = np.array(np.sum(Lambda, axis=0)).reshape(-1)
    Psbm = np.einsum("ij,i->ij", Lambda, 1 / np.maximum(1, np.sum(Lambda, axis=1)))
    return Din, Psbm, U
