"""Modules for graph embedding methods."""

import logging

import gensim
import networkx as nx
import numpy as np
import pandas as pd

# For GCN
import stellargraph as sg
import tensorflow as tf
from graph_embeddings import samplers, utils
from scipy import sparse
from sklearn import model_selection
from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

try:
    import glove
except ImportError:
    print(
        "Ignore this message if you do not use Glove. Otherwise, install glove python package by 'pip install glove_python_binary' "
    )


#
# Base class
#
class NodeEmbeddings:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass


class Node2Vec(NodeEmbeddings):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 5)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
    restart_prob : float (optional, default 0)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=5,
        walk_length=40,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
        random_teleport=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.sampler = samplers.SimpleWalkSampler(
            num_walks,
            walk_length,
            window_length,
            restart_prob,
            p,
            q,
            sample_center_context_pairs=False,
            verbose=False,
            random_teleport=random_teleport,
        )

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": 1,
            "workers": 4,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.sampler.window_length

        self.sentences = utils.walk2gensim_sentence(
            self.sampler.walks, self.sampler.window_length
        )

        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(sentences=self.sentences, **self.w2vparams)

        num_nodes = len(self.model.wv.key_to_index)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if "%d" % i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv["%d" % i]
            self.out_vec[i, :] = self.model.syn1neg[
                self.model.wv.key_to_index["%d" % i]
            ]


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams["sg"] = 0
        self.w2vparams["hs"] = 1


class Glove:
    def __init__(
        self,
        num_walks=5,
        walk_length=40,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.sampler = samplers.SimpleWalkSampler(
            num_walks,
            walk_length,
            window_length,
            restart_prob,
            p,
            q,
            sample_center_context_pairs=True,
            verbose=False,
        )
        self.learning_rate = 0.05
        self.w2vparams = {"epochs": 25, "no_threads": 4}

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        center, context, freq = self.sampler.get_center_context_pairs()
        center = center.astype(int)
        context = context.astype(int)
        N = self.sampler.num_nodes
        self.cooccur = sparse.coo_matrix(
            (freq, (center, context)), shape=(N, N), dtype="double"
        )
        return self

    def transform(self, dim, return_out_vector=False):
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that
        # for the previous call of transform function
        update_embedding = False
        if self.out_vec is None:
            update_embedding = True
        elif self.out_vec.shape[1] != dim:
            update_embedding = True

        # Update the dimension and train the model
        if update_embedding:
            self.model = glove.Glove(
                no_components=dim, learning_rate=self.learning_rate
            )
            self.model.fit(self.cooccur, **self.w2vparams)
            self.in_vec = self.model.word_vectors
            self.out_vec = self.model.word_vectors

        if return_out_vector:
            return self.out_vec
        else:
            return self.in_vec


class Fairwalk(Node2Vec):
    def __init__(self, group_membership=None, **params):
        Node2Vec.__init__(self, **params)
        self.group_membership = group_membership
        self.w2vparams = {
            "sg": 0,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)

        if self.group_membership is None:  # default is degree
            self.group_membership = np.unique(
                np.array(A.sum(axis=1)).reshape(-1), return_inverse=True
            )[1]

        # balance transition probability
        Ahat = A.copy()
        num_nodes = A.shape[0]
        for i in range(num_nodes):
            # Compute the out-deg
            node_ids = A.indices[A.indptr[i] : A.indptr[i + 1]]
            _, gids, freq = np.unique(
                self.group_membership[node_ids], return_inverse=True, return_counts=True
            )
            w = 1 / freq[gids]

            Ahat.data[A.indptr[i] : A.indptr[i + 1]] = w
        self.sampler.sampling(Ahat)
        return self


class LaplacianEigenMap(NodeEmbeddings):
    def __init__(self):
        self.in_vec = None
        self.L = None
        self.deg = None

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        u, s, _ = utils.rSVD(self.L, dim + 1)  # add one for the trivial solution
        order = np.argsort(s)[::-1][1:]
        u = u[:, order]

        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
        self.in_vec = Dsqrt @ u
        self.out_vec = u


class NetMF(NodeEmbeddings):
    """NetMF.

    Alias of LevyWord2Vec
    """

    def __init__(self, window_length=10, num_neg_samples=1, h=256, **params):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.b = num_neg_samples
        self.h = h

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        if self.h is None:
            self.h = np.power(net.shape[0], 0.66)
        if self.h > net.shape[0]:
            self.h = net.shape[0]

        if self.window_length > 4:
            logger.debug("Approximiating Mhat")
            deg = np.array(net.sum(axis=0)).reshape(-1)
            Dsqinv = sparse.diags(np.sqrt(1 / np.maximum(deg, 1)))
            Uh, Lamh, Vh = utils.rSVD(Dsqinv @ net @ Dsqinv, self.h)
            S = np.sign(Lamh)
            # out_vec = np.einsum("ij,i->ij", Vh, S)
            Lamh = Lamh * S

            averaged_Lamh = Lamh.copy()
            for t in range(2, self.window_length):
                averaged_Lamh += np.power(Lamh, t)
            averaged_Lamh /= self.window_length

            logger.debug("Computing Mhat")
            Uh = Dsqinv @ Uh @ sparse.diags(np.sqrt(averaged_Lamh))
            self.Mhat = net.sum() * (Uh @ Uh.T) / self.b
        else:
            deg = np.array(net.sum(axis=0)).reshape(-1)
            self.Mhat = utils.calc_rwr(utils.to_trans_mat(net), 0, self.window_length)
            self.Mhat = self.Mhat @ np.diag(1 / np.maximum(deg, 1)) * np.sum(deg)

        logger.debug("Thresholding")
        self.Mhat = np.log(np.maximum(self.Mhat, 1))
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model

        # Sample the sequence of nodes using a random walk
        logger.debug("SVD")
        in_vec, val, out_vec = utils.rSVD(self.Mhat, dim)
        # in_vec, val, out_vec = sparse.linalg.svds(self.Mhat, dim)
        # in_vec, val, out_vec = utils.rSVD(self.Mhat + sparse.diags(np.ones()), dim)
        order = np.argsort(val)[::-1]
        val = val[order]
        alpha = 0.5
        self.in_vec = in_vec[:, order] @ np.diag(np.power(val, alpha))
        self.out_vec = out_vec[order, :].T @ np.diag(np.power(val, 1 - alpha))


class GAT(NodeEmbeddings):
    """A python class for GAT."""

    def __init__(
        self,
        number_of_walks=1,
        batch_size=512,
        epochs=200,
        lr=1e-2,
        num_samples=[25, 10],
        layer_sizes=[256, 256],
        num_default_features=None,
    ):

        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

        self.number_of_walks = number_of_walks
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.num_default_features = num_default_features

    def fit(self, net, node_features=None):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """

        def find_blocks_by_sbm(A, K, directed=False):
            """Jiashun Jin. Fast community detection by SCORE.

            :param A: scipy sparse matrix
            :type A: sparse.csr_matrix
            :param K: number of communities
            :type K: int
            :param directed: whether to cluster directed or undirected, defaults to False
            :type directed: bool, optional
            :return: [description]
            :rtype: [type]
            """

            if K >= (A.shape[0] - 1):
                cids = np.arange(A.shape[0])
                return cids
            u, s, v = utils.rSVD(A, dim=K)

            u = np.ascontiguousarray(u, dtype=np.float32)
            if directed:
                v = np.ascontiguousarray(v.T, dtype=np.float32)
                u = np.hstack([u, v])
            norm = np.linalg.norm(u, axis=1)
            denom = 1 / np.maximum(norm, 1e-5)
            denom[np.isclose(norm, 0)] = 0

            u = np.einsum("ij,i->ij", u, denom)

            if (u.shape[0] / K) < 10:
                niter = 1
            else:
                niter = 10
            km = faiss.Kmeans(u.shape[1], K, niter=niter, spherical=True)
            km.train(u)
            _, cids = km.index.search(u, 1)
            cids = np.array(cids).reshape(-1)

            return np.array(cids).reshape(-1)

        logger.debug("sampling - start")
        A = utils.to_adjacency_matrix(net)
        Gnx = nx.from_scipy_sparse_matrix(A)

        deg = np.array(A.sum(axis=1)).reshape(-1)
        self.deg = np.array([deg[i] for i in Gnx.nodes])
        self.Gnx = Gnx
        self.num_nodes = len(Gnx.nodes)
        self.A = A
        self.node_features = node_features
        return self

    def update_embedding(self, dim):
        self.layer_sizes[-1] = dim

        if self.node_features is None:
            if self.num_default_features is None:
                self.num_default_features = dim
            d = np.maximum(1, np.array(self.A.sum(axis=1)).reshape(-1))
            dsqrt = np.sqrt(d)
            L = sparse.diags(1 / dsqrt) @ self.A @ sparse.diags(1 / dsqrt)
            X, s, _ = utils.rSVD(L, self.num_default_features)
            node_features = pd.DataFrame(X)
            node_features["deg"] = self.deg
            X = node_features.values
            X = X @ np.diag(1 / np.maximum(np.linalg.norm(X, axis=0), 1e-12))
            node_features = pd.DataFrame(X)
        else:
            node_features = self.node_features

        self.G = sg.StellarGraph.from_networkx(self.Gnx, node_features=node_features)
        self.train_targets, self.test_targets = model_selection.train_test_split(
            node_features, train_size=0.5
        )
        self.val_targets, self.test_targets = model_selection.train_test_split(
            self.test_targets, train_size=0.5, test_size=None
        )

        generator = sg.mapper.FullBatchNodeGenerator(self.G, method="gat")
        gat = sg.layer.GAT(
            layer_sizes=self.layer_sizes,
            # layer_sizes=[8, train_targets.shape[1]],
            activations=["elu", "softmax"],
            attn_heads=8,
            in_dropout=0.5,
            attn_dropout=0.5,
            normalize=None,
            generator=generator,
        )
        x_inp, x_out = gat.in_out_tensors()

        predictions = tf.keras.layers.Dense(
            units=self.train_targets.shape[1]  # , activation="linear"
        )(x_out)
        model = tf.keras.Model(inputs=x_inp, outputs=predictions)

        model.compile(
            keras.optimizers.Adam(lr=self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
            # metrics=["accuracy"],
        )

        all_gen = generator.flow(np.arange(self.num_nodes))
        train_gen = generator.flow(self.train_targets.index, self.train_targets)
        val_gen = generator.flow(self.val_targets.index, self.val_targets)

        es_callback = EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True
        )
        model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback],
        )
        embedding_model = Model(inputs=x_inp, outputs=x_out)
        emb = embedding_model.predict(all_gen).reshape((self.num_nodes, dim))
        # %%
        self.in_vec = emb.copy()
        self.out_vec = emb.copy()
        return self


class GCN(NodeEmbeddings):
    """A python class for GCN."""

    def __init__(
        self,
        number_of_walks=1,
        batch_size=512,
        epochs=200,
        lr=1e-2,
        num_samples=[25, 10],
        layer_sizes=[256, 256],
        num_default_features=None,
    ):

        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

        self.number_of_walks = number_of_walks
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.import_lib()
        self.num_default_features = num_default_features

    def import_lib(self):

        import stellargraph as sg
        import tensorflow as tf
        from sklearn import model_selection, preprocessing
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from stellargraph.data import UnsupervisedSampler
        from stellargraph.layer import GraphSAGE, link_classification
        from stellargraph.mapper import (
            FullBatchNodeGenerator,
            GraphSAGELinkGenerator,
            GraphSAGENodeGenerator,
        )
        from tensorflow import keras
        from tensorflow.keras import Model, layers, losses, metrics, optimizers
        from tensorflow.keras.callbacks import EarlyStopping

    def fit(self, net, node_features=None):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """

        def find_blocks_by_sbm(A, K, directed=False):
            """Jiashun Jin. Fast community detection by SCORE.

            :param A: scipy sparse matrix
            :type A: sparse.csr_matrix
            :param K: number of communities
            :type K: int
            :param directed: whether to cluster directed or undirected, defaults to False
            :type directed: bool, optional
            :return: [description]
            :rtype: [type]
            """

            if K >= (A.shape[0] - 1):
                cids = np.arange(A.shape[0])
                return cids
            u, s, v = utils.rSVD(A, dim=K)

            u = np.ascontiguousarray(u, dtype=np.float32)
            if directed:
                v = np.ascontiguousarray(v.T, dtype=np.float32)
                u = np.hstack([u, v])
            norm = np.linalg.norm(u, axis=1)
            denom = 1 / np.maximum(norm, 1e-5)
            denom[np.isclose(norm, 0)] = 0

            u = np.einsum("ij,i->ij", u, denom)

            if (u.shape[0] / K) < 10:
                niter = 1
            else:
                niter = 10
            km = faiss.Kmeans(u.shape[1], K, niter=niter, spherical=True)
            km.train(u)
            _, cids = km.index.search(u, 1)
            cids = np.array(cids).reshape(-1)

            return np.array(cids).reshape(-1)

        logger.debug("sampling - start")
        A = utils.to_adjacency_matrix(net)
        Gnx = nx.from_scipy_sparse_matrix(A)

        deg = np.array(A.sum(axis=1)).reshape(-1)
        self.deg = np.array([deg[i] for i in Gnx.nodes])
        self.Gnx = Gnx
        self.num_nodes = len(Gnx.nodes)
        self.A = A
        self.node_features = node_features
        return self

    def update_embedding(self, dim):
        self.layer_sizes[-1] = dim

        if self.node_features is None:
            if self.num_default_features is None:
                self.num_default_features = dim
            d = np.maximum(1, np.array(self.A.sum(axis=1)).reshape(-1))
            dsqrt = np.sqrt(d)
            L = sparse.diags(1 / dsqrt) @ self.A @ sparse.diags(1 / dsqrt)
            X, s, _ = utils.rSVD(L, self.num_default_features)
            node_features = pd.DataFrame(X)
            node_features["deg"] = self.deg
            X = node_features.values
            X = X @ np.diag(1 / np.maximum(np.linalg.norm(X, axis=0), 1e-12))
            node_features = pd.DataFrame(X)
        else:
            node_features = self.node_features

        self.G = sg.StellarGraph.from_networkx(self.Gnx, node_features=node_features)
        self.train_targets, self.test_targets = model_selection.train_test_split(
            node_features, train_size=0.5
        )
        self.val_targets, self.test_targets = model_selection.train_test_split(
            self.test_targets, train_size=0.5, test_size=None
        )

        generator = sg.mapper.FullBatchNodeGenerator(self.G, method="gcn")
        gcn = sg.layer.GCN(
            layer_sizes=self.layer_sizes,
            generator=generator,
            activations=["relu", "relu"],
            dropout=0.5,
        )
        x_inp, x_out = gcn.in_out_tensors()

        predictions = tf.keras.layers.Dense(
            units=self.train_targets.shape[1]  # , activation="linear"
        )(x_out)
        model = tf.keras.Model(inputs=x_inp, outputs=predictions)

        model.compile(
            keras.optimizers.Adam(lr=self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
            # metrics=["accuracy"],
        )

        all_gen = generator.flow(np.arange(self.num_nodes))
        train_gen = generator.flow(self.train_targets.index, self.train_targets)
        val_gen = generator.flow(self.val_targets.index, self.val_targets)

        es_callback = EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True
        )
        model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            callbacks=[es_callback],
        )
        embedding_model = Model(inputs=x_inp, outputs=x_out)
        emb = embedding_model.predict(all_gen).reshape((self.num_nodes, dim))
        # %%
        self.in_vec = emb.copy()
        self.out_vec = emb.copy()
        return self


class GraphSage(GCN):
    """A python class for GCN."""

    def __init__(
        self,
        length=2,
        number_of_walks=1,
        batch_size=512,
        epochs=5,
        lr=1e-2,
        num_samples=[25, 10],
        layer_sizes=[256, 256],
        num_default_features=None,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

        self.length = length
        self.number_of_walks = number_of_walks
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.import_lib()
        self.num_default_features = num_default_features

    def fit(self, net, node_features=None):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """

        def find_blocks_by_sbm(A, K, directed=False):
            """Jiashun Jin. Fast community detection by SCORE.

            :param A: scipy sparse matrix
            :type A: sparse.csr_matrix
            :param K: number of communities
            :type K: int
            :param directed: whether to cluster directed or undirected, defaults to False
            :type directed: bool, optional
            :return: [description]
            :rtype: [type]
            """

            if K >= (A.shape[0] - 1):
                cids = np.arange(A.shape[0])
                return cids
            u, s, v = utils.rSVD(A, dim=K)

            u = np.ascontiguousarray(u, dtype=np.float32)
            if directed:
                v = np.ascontiguousarray(v.T, dtype=np.float32)
                u = np.hstack([u, v])
            norm = np.linalg.norm(u, axis=1)
            denom = 1 / np.maximum(norm, 1e-5)
            denom[np.isclose(norm, 0)] = 0

            u = np.einsum("ij,i->ij", u, denom)

            if (u.shape[0] / K) < 10:
                niter = 1
            else:
                niter = 10
            km = faiss.Kmeans(u.shape[1], K, niter=niter, spherical=True)
            km.train(u)
            _, cids = km.index.search(u, 1)
            cids = np.array(cids).reshape(-1)

            return np.array(cids).reshape(-1)

        logger.debug("sampling - start")
        A = utils.to_adjacency_matrix(net)
        Gnx = nx.from_scipy_sparse_matrix(A)

        deg = np.array(A.sum(axis=1)).reshape(-1)
        self.deg = np.array([deg[i] for i in Gnx.nodes])
        self.Gnx = Gnx
        self.A = A
        self.node_features = node_features

        return self

    def update_embedding(self, dim):

        if self.node_features is None:
            if self.num_default_features is None:
                self.num_default_features = dim

            d = np.maximum(1, np.array(self.A.sum(axis=1)).reshape(-1))
            dsqrt = np.sqrt(d)
            L = sparse.diags(1 / dsqrt) @ self.A @ sparse.diags(1 / dsqrt)
            X, s, _ = utils.rSVD(L, self.num_default_features)
            node_features = pd.DataFrame(X)
            node_features["deg"] = self.deg
            X = node_features.values
            X = X @ np.diag(1 / np.maximum(np.linalg.norm(X, axis=0), 1e-12))
            node_features = pd.DataFrame(X)
        else:
            node_features = self.node_features

        self.G = sg.StellarGraph.from_networkx(
            self.Gnx, node_features=node_features
        )  # node_features)

        unsupervised_samples = UnsupervisedSampler(
            self.G,
            nodes=list(self.Gnx.nodes),
            length=self.length,
            number_of_walks=self.number_of_walks,
        )
        generator = GraphSAGELinkGenerator(self.G, self.batch_size, self.num_samples)
        train_gen = generator.flow(unsupervised_samples)

        self.layer_sizes[-1] = dim
        graphsage = GraphSAGE(
            layer_sizes=self.layer_sizes,
            generator=generator,
            bias=True,
            dropout=0.0,
            normalize="l2",
        )
        x_inp, x_out = graphsage.in_out_tensors()
        prediction = link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
        )(x_out)
        model = keras.Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.binary_accuracy],
        )
        model.fit(
            train_gen,
            epochs=self.epochs,
            verbose=1,
            use_multiprocessing=False,
            workers=3,
            shuffle=True,
        )
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

        # %%
        node_gen = GraphSAGENodeGenerator(
            self.G, self.batch_size, self.num_samples
        ).flow(list(self.Gnx.nodes))
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

        # %%
        self.in_vec = node_embeddings.copy()
        self.out_vec = node_embeddings.copy()
        return self
