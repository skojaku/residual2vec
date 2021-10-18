import csv
import logging
from os.path import join as osjoin

import networkx as nx
import numpy as np
from link_prediction.utils import mk_outdir
from scipy import sparse
from scipy.sparse import csgraph
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class NetworkTrainTestSplitter(object):
    """Train and Test set splitter for network data.

    This class is for link prediction tasks.
    """

    def __init__(self, G, directed=False, fraction=0.5):
        """
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        self.G = G
        self.directed = directed

        self.original_edge_set = set(G.edges)
        self.node_list = list(G.nodes)
        self.total_number_of_edges = len(self.original_edge_set)
        self.number_of_test_edges = int(self.total_number_of_edges * fraction)

        self.test_edges = []
        self.negative_edges = []

    def train_test_split(self):
        """Split train and test edges.

        Train network should have a one weakly connected component.
        """
        logging.info("Initiate train test set split")
        check_connectivity_method = (
            nx.is_weakly_connected if self.directed else nx.is_connected
        )
        while len(self.test_edges) != self.number_of_test_edges:
            edge_list = np.array(self.G.edges())
            candidate_idxs = np.random.choice(
                len(edge_list),
                self.number_of_test_edges - len(self.test_edges),
                replace=False,
            )
            for source, target in tqdm(edge_list[candidate_idxs]):
                self.G.remove_edge(source, target)
                if check_connectivity_method(self.G):
                    self.test_edges.append((source, target))
                else:
                    self.G.add_edge(source, target, weight=1)

    def generate_negative_edges(self):
        """Generate a negative samples for link prediction task."""
        logging.info("Initiate generating negative edges")

        # Map a pair of integers into a single integer
        def pairing(i, j, N):
            return np.minimum(i, j) + np.maximum(i, j) * N

        # Map an integer to a pair of integers
        def depairing(i, N):
            i, j = divmod(i, N)
            return np.minimum(i, j), np.maximum(i, j)

        # Compute the pair ids for adjacent nodes
        N = self.G.number_of_nodes()
        nodelist = np.array(list(self.node_list))
        node2id = dict(zip(nodelist, np.arange(len(nodelist))))
        p_edges = set(
            [pairing(node2id[r], node2id[c], N) for r, c in self.original_edge_set]
        )

        # Compute the ids for self-loops
        self_edges = set(pairing(np.arange(N), np.arange(N), N))

        # Set of negative edges
        n_edges = set([])

        while len(n_edges) < self.number_of_test_edges:

            # Random sample node pairs
            remaining = self.number_of_test_edges - len(n_edges)
            source = np.random.choice(N, remaining, replace=True)
            target = np.random.choice(N, remaining, replace=True)

            # Pairing the sampled nodes
            sampled_edges = pairing(source, target, N)

            # Remove node pairs that are already taken or not allowed
            sampled_edges = set(sampled_edges) - n_edges - p_edges - self_edges

            # Add the sampled edges to the negative edges
            n_edges = n_edges.union(sampled_edges)

        # Depair edge ids to pairs of node ids
        source, target = depairing(np.array(list(n_edges)), N)

        # Map node ids to node label
        source, target = nodelist[source], nodelist[target]
        self.negative_edges = list(tuple(zip(source, target)))

    def save_splitted_result(self, path):
        """
        :param path: path for saving result files (train network, test_edges, negative edges)
        """
        mk_outdir(path)
        nx.write_edgelist(self.G, osjoin(path, "network.elist"))
        with open(osjoin(path, "test_edges.tsv"), "wt") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerows(self.test_edges)
        with open(osjoin(path, "negative_edges.tsv"), "wt") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerows(self.negative_edges)
        logging.info("Train-test splitter data are stored")


class NetworkTrainTestSplitterFast(NetworkTrainTestSplitter):
    def __init__(self, G, directed=False, fraction=0.5):
        """Only support undirected Network.

        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        if nx.is_directed(G):
            raise NotImplementedError
        super(NetworkTrainTestSplitterFast, self).__init__(
            G, directed=directed, fraction=fraction
        )

    def train_test_split(self):
        """Split train and test edges.

        Train network should have a one weakly connected component.
        """
        logging.info("Initiate train test set split with fast version")

        while len(self.test_edges) != self.number_of_test_edges:
            edge_list = np.array(self.G.edges())
            candidate_idxs = np.random.choice(
                len(edge_list),
                self.number_of_test_edges - len(self.test_edges),
                replace=False,
            )
            for source, target in tqdm(edge_list[candidate_idxs]):
                # cases sure cannot remove the edge:
                # one node is dangling
                if self.G.degree(source) == 1 or self.G.degree(target) == 1:
                    continue

                self.G.remove_edge(source, target)
                # cases sure can remove the edge:
                # source is reachable for target through other nodes
                # instead of using the default check connectivity method
                # here we use a lazy BFS method to stop early if target is reachable
                reachable = False
                seen = {}  # level (number of hops) when seen in BFS
                level = 0  # the current level
                nextlevel = {source: 1}  # dict of nodes to check at next level
                while nextlevel:
                    thislevel = nextlevel  # advance to next level
                    nextlevel = {}  # and start a new list (fringe)
                    for v in thislevel:
                        if v not in seen:
                            seen[v] = level  # set the level of vertex v
                            nextlevel.update(self.G[v])  # add neighbors of v
                    if target in seen:
                        reachable = True
                        break
                    level = level + 1

                if reachable:
                    self.test_edges.append((source, target))
                else:
                    self.G.add_edge(source, target, weight=1)


class NetworkTrainTestSplitterWithMST(NetworkTrainTestSplitter):
    def __init__(self, G, directed=False, fraction=0.5):
        """Only support undirected Network.

        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        if nx.is_directed(G):
            raise NotImplementedError
        super(NetworkTrainTestSplitterWithMST, self).__init__(
            G, directed=directed, fraction=fraction
        )

    def find_mst(self, G):
        MST = csgraph.minimum_spanning_tree(nx.adjacency_matrix(G))
        MST = nx.from_scipy_sparse_matrix(MST)
        nodes = np.array(list(G.nodes()))
        mapping = dict(zip(np.arange(len(nodes)), nodes))
        return nx.relabel_nodes(MST, mapping)

    def train_test_split(self):
        """Split train and test edges with MST.

        Train network should have a one weakly connected component.
        """
        logging.info("Initiate train test set split with MST")

        MST = self.find_mst(self.G)
        remained_edge_set = np.array(list(set(self.G.edges()) - set(MST.edges())))
        if len(remained_edge_set) < self.number_of_test_edges:
            raise Exception(
                "Cannot remove edges by keeping the connectedness. Decrease the `fraction` parameter"
            )

        edge_ids = np.random.choice(
            len(remained_edge_set), self.number_of_test_edges, replace=False
        )
        self.test_edges = remained_edge_set[edge_ids]
        self.G.remove_edges_from(self.test_edges)
