import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import residual2vec as rv


class TestResidual2Vec(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_residual2vec(self):
        model = rv.Residual2Vec()
        model.fit(self.A)
        vec = model.transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]


if __name__ == "__main__":
    unittest.main()
