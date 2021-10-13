import shutil
import unittest

import graph_embeddings
import networkx as nx
import numpy as np
from scipy import sparse


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_node2vec(self):
        model = graph_embeddings.Node2Vec()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_deepwalk(self):
        model = graph_embeddings.DeepWalk()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_glove(self):
        model = graph_embeddings.Glove()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_fairwalk(self):
        model = graph_embeddings.Fairwalk()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_laplacian_eigen_map(self):
        model = graph_embeddings.LaplacianEigenMap()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_netmf(self):
        model = graph_embeddings.NetMF()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_gat(self):
        model = graph_embeddings.GAT()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_gcn(self):
        model = graph_embeddings.GCN()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_graphsage(self):
        model = graph_embeddings.GraphSage()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32


if __name__ == "__main__":
    unittest.main()
