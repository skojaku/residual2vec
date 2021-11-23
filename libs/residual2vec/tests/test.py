import unittest

import networkx as nx
import numpy as np

import residual2vec as rv


class TestResidual2Vec(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_mf_fit_and_transform(self):
        model = rv.residual2vec_matrix_factorization()
        model.fit(self.A)
        vec = model.transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_mf_fit_transform(self):
        model = rv.residual2vec_matrix_factorization()
        vec = model.fit(self.A).transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_mf_random_graphs(self):
        group_membership = np.random.choice(3, self.A.shape[0], replace=True)
        model = rv.residual2vec_matrix_factorization(group_membership=group_membership)
        vec = model.fit(self.A).transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_sgd_fit_and_transform(self):
        noise_sampler = rv.ConfigModelNodeSampler()
        model = rv.residual2vec_sgd(noise_sampler)
        model.fit(self.A)
        vec = model.transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_sgd_fit_transform(self):
        noise_sampler = rv.ConfigModelNodeSampler()
        model = rv.residual2vec_sgd(noise_sampler)
        vec = model.fit(self.A).transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]

    def test_sgd_random_graphs(self):
        group_membership = np.random.choice(3, self.A.shape[0], replace=True)
        noise_sampler = rv.SBMNodeSampler(group_membership=group_membership)
        model = rv.residual2vec_sgd(noise_sampler)
        vec = model.fit(self.A).transform(dim=32)

        assert vec.shape[1] == 32
        assert vec.shape[0] == self.A.shape[0]


if __name__ == "__main__":
    unittest.main()
