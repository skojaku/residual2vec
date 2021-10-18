"""Module for simulating random walks."""
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from graph_embeddings import utils
from scipy import sparse

logger = logging.getLogger(__name__)


class NodeSampler(metaclass=ABCMeta):
    """Super class for node sampler class.

    Implement
        - get_trans_matrix
        - sampling

    Optional
        - get_decomposed_trans_matrix
    """

    @abstractmethod
    def get_trans_matrix(self, scale="normal"):
        """Construct the transition matrix for the node sequence.

        Return
        ------
        trans_prob : sparse.csr_matrix
            Transition matrix. trans_prob[i,j] is the probability
            that a random walker in node i moves to node j.
        """

    def get_trans_prob(self, src, trg):
        """Construct the transition matrix for the node sequence.

        Parameters
        ----------
        src : numpy.ndarray
            IDs of source nodes
        trg : numpy.ndarray
            IDs of target nodes


        Return
        ------
        trans_prob : numpy.ndarray
            Transition probability
        """
        P = self.get_trans_matrix()
        return np.array(P[(src, trg)]).reshape(-1)

    @abstractmethod
    def sampling(self):
        """Generate a sequence of walks over the network.

        Return
        ------
        walks : numpy.ndarray (number_of_walks, number_of_steps)
            Each row indicates a trajectory of a random walker
            walk[i,j] indicates the jth step for walker i.
        """

    @abstractmethod
    def get_center_context_pairs(self, num_walks=5):
        """get center and context pairs."""

    @abstractmethod
    def sample_context(self, pos_pairs, sz):
        """sample context from center."""

    @abstractmethod
    def count_center_context_pairs(self):
        """count the number of possible pairs."""

    @abstractmethod
    def get_batch_pairs(self, batch_size):
        """get batch of center context pairs."""

    def get_decomposed_trans_matrix(self, scale="normal"):
        return [[self.get_trans_matrix(scale)]]


#
# SimpleWalk Sampler
#
class SimpleWalkSampler(NodeSampler):
    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
        sample_center_context_pairs=True,
        random_teleport=False,
        **params
    ):
        """Simple walk with restart.

        Parameters
        ----------
        num_walks : int (Optional; Default 1)
            Number of walkers to simulate for each randomized network.
            A larger value removes the bias better but takes more time.
        walk_length : int
            Number of steps for a single walker
        p : float
            Parameter for the node2vec
        q : float
            Parameter for the node2vec
        window_length : int
            Size of the window
        restart_prob : float
            Probability of teleport back to the starting node
        verbose : bool
            Set true to display the progress (NOT IMPLEMENTED)
        """

        self.restart_prob = restart_prob
        self.trans_prob = None
        self.verbose = verbose
        self.num_nodes = -1

        # parameters for random walker
        self.num_walks = int(num_walks)
        self.walk_length = walk_length
        self.sample_center_context_pairs = sample_center_context_pairs
        self.p = p
        self.q = q
        self.walks = None
        self.W = None
        self.cum_trans_prob = None

        # parameters for learning embeddings
        self.window_length = window_length
        self.random_teleport = random_teleport

    def _init_center_context_pairs(self, center, context, freq):
        self.W = sparse.csr_matrix(
            (freq, (center, context)), shape=(self.num_nodes, self.num_nodes),
        )
        self.cum_trans_prob = utils.calc_cum_trans_prob(self.W.copy())
        self.center_prob = np.array(self.W.sum(axis=1)).reshape(-1)
        if ~np.isclose(np.sum(self.center_prob), 0):
            self.center_prob /= np.sum(self.center_prob)

    def sampling(self, net):
        self.num_nodes = net.shape[0]
        self.A = net

        if np.isclose(self.p, 1) and np.isclose(self.q, 1):
            if self.sample_center_context_pairs:
                center, context, freq = utils.sample_center_context_pair(
                    self.A,
                    self.num_walks,
                    self.walk_length,
                    self.restart_prob,
                    self.window_length,
                    random_teleport=self.random_teleport,
                )
                self._init_center_context_pairs(center, context, freq)
            else:
                self.walks = utils.simulate_simple_walk(
                    self.A,
                    self.num_walks,
                    self.walk_length,
                    self.restart_prob,
                    random_teleport=self.random_teleport,
                )
                self.walks = self.walks.astype(int)
        else:  # biased walk
            # calc supra-adjacency matrix
            Asupra, node_pairs = utils.construct_line_net_adj(net, p=self.p, q=self.q)
            self.num_nodes_supra = Asupra.shape[0]

            # Find the starting node ids
            start_node_ids = np.where(node_pairs[:, 0] == node_pairs[:, 1])[0]
            self.walks = utils.simulate_simple_walk(
                Asupra,
                self.num_walks,
                self.walk_length,
                self.restart_prob,
                start_node_ids=start_node_ids,
            )
            self.walks = self.walks.astype(int)

            # Convert supra-node id to the id of the nodes in the original net
            self.walks = node_pairs[self.walks.reshape(-1), 1].reshape(self.walks.shape)

    def get_trans_matrix(self, scale="normal"):
        """Compute the transition probability from one node to another.

        Return
        ------
        trans_prob : numpy.ndarray
            Transition probability matrix of size.
            (number of nodes, number of nodes).
            trans_prob[i,j] indicates the transition probability
            from node i to node j.
        """

        # Generate a sequence of random walks
        if self.W is None:
            self.get_center_context_pairs()
        if self.joint_prob is False:
            trans_prob = utils.to_trans_mat(self.W)
        if scale == "log":
            trans_prob.data = utils.safe_log(trans_prob.data)
        return trans_prob

    def get_center_context_pairs(self):
        if self.W is None:
            (center, context, freq,) = utils.generate_center_context_pair(
                self.walks, self.window_length
            )
            self._init_center_context_pairs(center, context, freq)
            return center, context, freq
        else:
            center, context, freq = sparse.find(self.W)
            return center, context, freq

    def count_center_context_pairs(self):
        return self.W.sum()

    def get_batch_pairs(self, batch_size):
        center = np.random.choice(self.num_nodes, batch_size, p=self.center_prob)
        context = self.sample_context(center, 1).reshape(-1)
        return center, context

    def sample_context(self, center, sz):
        context = utils.sample_columns_from_cum_prob(
            np.repeat(center, sz),
            self.cum_trans_prob.indptr,
            self.cum_trans_prob.indices,
            self.cum_trans_prob.data,
        )
        context = context.reshape((len(center), sz))
        return context
