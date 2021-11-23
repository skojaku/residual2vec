"""Graph module to store a network and generate random walks from it."""
import numpy as np
from scipy import sparse

from residual2vec import utils


class NodeSampler:
    def fit(self, A):
        """Fit the sampler.

        :param A: adjacency matrix
        :type A: scipy.csr_matrix
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError

    def sampling(self, center_node, n_samples):
        """Sample context nodes from the graph for center nodes.

        :param center_node: ID of center node
        :type center_node: int
        :param n_samples: number of samples per center node
        :type n_samples: int
        :raises NotImplementedError: [description]
        """
        raise NotImplementedError


class SBMNodeSampler(NodeSampler):
    """Node Sampler based on the stochatic block model."""

    def __init__(
        self, window_length=10, group_membership=None, dcsbm=True,
    ):
        """Node Sampler based on the stochatic block model.

        :param window_length: length of the context window, defaults to 10
        :type window_length: int, optional
        :param group_membership: group membership of nodes, defaults to None
        :type group_membership: np.ndarray, optional
        :param dcsbm: Set dcsbm=True to take into account the degree of nodes, defaults to True
        :type dcsbm: bool, optional
        """
        if group_membership is None:
            self.group_membership = None
        else:
            self.group_membership = np.unique(group_membership, return_inverse=True)[
                1
            ]  # reindex
        self.window_length = window_length
        self.dcsbm = dcsbm

    def fit(self, A):
        """Initialize the dcSBM sampler."""

        # Initalize the parameters
        self.n_nodes = A.shape[0]

        # Initialize the group membership
        if self.group_membership is None:
            self.group_membership = np.zeros(self.n_nodes, dtype=np.int64)
            self.node2group = utils.to_member_matrix(self.group_membership)
        else:
            self.node2group = utils.to_member_matrix(self.group_membership)

        indeg = np.array(A.sum(axis=0)).reshape(-1)
        Lambda = (self.node2group.T @ A @ self.node2group).toarray()
        Din = np.array(np.sum(Lambda, axis=0)).reshape(-1)
        Nin = np.array(self.node2group.sum(axis=0)).reshape(-1)
        Psbm = np.einsum(
            "ij,i->ij", Lambda, 1 / np.maximum(1, np.array(np.sum(Lambda, axis=1)))
        )
        Psbm_pow = utils.matrix_sum_power(Psbm, self.window_length) / self.window_length

        if self.dcsbm:
            self.block2node = (
                sparse.diags(1 / np.maximum(1, Din))
                @ sparse.csr_matrix(self.node2group.T)
                @ sparse.diags(indeg)
            )
        else:
            self.block2node = sparse.diags(1 / np.maximum(1, Nin)) @ sparse.csr_matrix(
                self.node2group.T
            )

        # From block to block
        self.block2block = sparse.csr_matrix(Psbm_pow)
        self.block2block.data = utils._csr_row_cumsum(
            self.block2block.indptr, self.block2block.data
        )

        # From block to node
        self.block2node.data = utils._csr_row_cumsum(
            self.block2node.indptr, self.block2node.data
        )

    def sampling(self, center_nodes, n_samples):
        _center_nodes = np.repeat(center_nodes, n_samples)
        block_ids = utils.csr_sampling(
            self.group_membership[_center_nodes], self.block2block
        )
        context = utils.csr_sampling(block_ids, self.block2node)
        return context.astype(np.int64).reshape((-1, n_samples))


class ConfigModelNodeSampler(SBMNodeSampler):
    def __init__(self):
        super(ConfigModelNodeSampler, self).__init__(window_length=1, dcsbm=True)


class ErdosRenyiNodeSampler(SBMNodeSampler):
    def __init__(self):
        super(ErdosRenyiNodeSampler, self).__init__(window_length=1, dcsbm=False)
