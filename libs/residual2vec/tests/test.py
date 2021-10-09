import unittest

import networkx as nx
import numpy as np

import residual2vec as rv

# from scipy import sparse


class TestResidual2Vec(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_fit_and_transform(self):
        model = rv.residual2vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_fit_transform(self):
        model = rv.residual2vec()
        vec = model.fit(self.A).transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_random_graphs(self):
        group_membership = np.random.choice(3, self.A.shape[0], replace=True)
        model = rv.residual2vec(group_membership=group_membership)
        vec = model.fit(self.A).transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]


if __name__ == "__main__":
    unittest.main()
