import logging
from collections import Counter

import networkx as nx
import numba
import numpy as np
from numba import prange
from scipy import sparse

logger = logging.getLogger(__name__)

#
# Random walks
#
def calc_cum_trans_prob(A):
    P = A.copy()
    a = _calc_cum_trans_prob(P.indptr, P.indices, P.data.astype(float), P.shape[0])
    P.data = a
    return P


# @numba.jit(nopython=True, parallel=True)
def _calc_cum_trans_prob(
    A_indptr, A_indices, A_data_, num_nodes,  # should be cumulative
):
    A_data = A_data_.copy()
    for i in range(num_nodes):
        # Compute the out-deg
        outdeg = np.sum(A_data[A_indptr[i] : A_indptr[i + 1]])
        A_data[A_indptr[i] : A_indptr[i + 1]] = np.cumsum(
            A_data[A_indptr[i] : A_indptr[i + 1]]
        ) / np.maximum(outdeg, 1)
    return A_data


@numba.jit(nopython=True, parallel=True)
def sample_columns_from_cum_prob(rows, A_indptr, A_indices, A_data):
    retvals = -np.ones(len(rows))
    for i in range(len(rows)):
        r = rows[i]
        nnz_row = A_indptr[r + 1] - A_indptr[r]
        if nnz_row == 0:
            continue

        # find a neighbor by a roulette selection
        _ind = np.searchsorted(
            A_data[A_indptr[r] : A_indptr[r + 1]], np.random.rand(), side="right",
        )
        retvals[i] = A_indices[A_indptr[r] + _ind]
    return retvals


def simulate_simple_walk(
    A,
    num_walk,
    walk_length,
    restart_prob,
    start_node_ids=None,
    restart_at_dangling=False,
    is_cum_trans_prob_mat=False,
    random_teleport=False,
):
    """Wrapper for."""

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    if start_node_ids is None:
        start_node_ids = np.arange(A.shape[0])

    # Extract the information on the csr matrix
    logger.debug(
        "simulate random walks: network of {} nodes. {} walkers per node. {} start nodes".format(
            A.shape[0], num_walk, len(start_node_ids)
        )
    )
    if is_cum_trans_prob_mat is False:
        logger.debug("Calculating the transition probability")
        P = calc_cum_trans_prob(A)
    else:
        P = A

    logger.debug("Start simulation")
    return _simulate_simple_walk(
        P.indptr,
        P.indices,
        P.data.astype(float),
        P.shape[0],
        float(restart_prob),
        np.repeat(start_node_ids, num_walk),
        walk_length,
        restart_at_dangling,
        random_teleport,
    )


# @numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True, cache=True, parallel=True)
def _simulate_simple_walk(
    A_indptr,
    A_indices,
    A_data,  # should be cumulative
    num_nodes,
    restart_prob,
    start_node_ids,
    walk_length,
    restart_at_dangling,
    random_teleport,
):
    """Sampler based on a simple walk. A random walker chooses a neighbor with
    proportional to edge weights. For a fast simulation of random walks, we
    exploit scipy.sparse.csr_matrix data structure. In scipy.sparse.csr_matrix
    A, there are 3 attributes:

    - A.indices : column ID
    - A.data : element value
    - A.indptr : the first index of A.indices for the ith row
    The neighbors of node i are given by A.indices[A.indptr[i]:A.indptr[i+1]], and
    the edge weights are given by A.data[A.indptr[i]:A.indptr[i+1]].
    Parameters
    ----------
    A: scipy.sparse.csr_matrix
        Adjacency matrix of a network
    walk_length: int
        Length of a single walk
    window_length: int
        Length of a window rolling over a generated sentence
    restart_prob: float (Optional; Default 0)
        Restart probability
    return_sentence bool (Optional; Default True)
        Return sentence in form of numpy.ndarray. Set false if using residual2vec
            Returns
    -------
    sampler: func
        A function that takes start node ids (numpy.ndarray) as input.
        If return_sentence is True, sampler returns sentences in form of numpy.ndarray
        where sentence[n,t] indicates the tth word in the nth sentence.
        If return_sentence is False, sampler returns center-context node pairs and its frequency:
        - center: numpy.ndarray
            Center words
        - context: numpy.ndarray
            Context words
        - freq: numpy.ndarray
            Frequency
    """
    # Alocate a memory for recording a walk
    walks = -np.ones((len(start_node_ids), walk_length), dtype=np.int32)
    for sample_id in prange(len(start_node_ids)):
        start_node = start_node_ids[sample_id]
        # Record the starting node
        visit = start_node
        walks[sample_id, 0] = visit
        for t in range(1, walk_length):
            # Compute the number of neighbors
            outdeg = A_indptr[visit + 1] - A_indptr[visit]
            # If reaches to an absorbing state, finish the walk
            # the random walker is teleported back to the starting node
            # or Random walk with restart
            if outdeg == 0:
                if restart_at_dangling:
                    if random_teleport:
                        next_node = np.random.randint(0, num_nodes)
                    else:
                        next_node = start_node
                else:
                    if t == 1:  # when starting from sink
                        pass
                        walks[sample_id, t] = visit
                    break
            elif np.random.rand() <= restart_prob:
                if random_teleport:
                    next_node = np.random.randint(0, num_nodes)
                else:
                    next_node = start_node
            else:
                # find a neighbor by a roulette selection
                _next_node = np.searchsorted(
                    A_data[A_indptr[visit] : A_indptr[visit + 1]],
                    np.random.rand(),
                    side="right",
                )
                next_node = A_indices[A_indptr[visit] + _next_node]
            # Record the transition
            walks[sample_id, t] = next_node
            # Move
            visit = next_node
    return walks


def sample_center_context_pair(
    A,
    num_walks,
    walk_length,
    restart_prob,
    window_length,
    random_teleport=False,
    batch_size=100000,
):

    num_nodes = A.shape[0]
    num_chunk = np.ceil(num_walks * num_nodes / batch_size).astype(int)
    pair_cnt = Counter()

    logger.debug("Calculating the transition probability")
    P = calc_cum_trans_prob(A)
    for ch_id, start_node_ids in enumerate(
        np.array_split(np.arange(num_nodes), num_chunk)
    ):
        logger.debug("chunk {} / {}: simulate random walk".format(ch_id, num_chunk))
        walks = simulate_simple_walk(
            P,
            num_walks,
            walk_length,
            restart_prob,
            start_node_ids=start_node_ids,
            is_cum_trans_prob_mat=True,
            random_teleport=random_teleport,
        )

        walk_length = walks.shape[1]
        walk_num = walks.shape[0]
        L_single = window_length * (walk_length - window_length)
        L_all = walk_num * L_single

        logger.debug(
            "chunk {} / {}: generating {} center context pairs".format(
                ch_id, num_chunk, L_all
            )
        )
        walks = walks.astype(int)
        pairs = _generate_center_context_pair_ids(walks, window_length)
        pair_cnt += Counter(pairs)
        logger.debug(
            "chunk {} / {}: {} pairs sampled in total".format(
                ch_id, num_chunk, len(pair_cnt)
            )
        )

    ids = np.fromiter(pair_cnt.keys(), dtype=int)
    freq = np.fromiter(pair_cnt.values(), dtype=float)
    w = np.floor((np.sqrt(8 * ids + 1) - 1) * 0.5)
    t = (w ** 2 + w) * 0.5
    context = ids - t
    center = w - context
    return center.astype(int), context.astype(int), freq


@numba.jit(nopython=True, cache=True, parallel=True)
def _generate_center_context_pair_ids(walks, window_length):

    """Generate center context node pairs from walks."""
    #
    # Allocate a memory for center and context node
    #
    walk_length = walks.shape[1]
    walk_num = walks.shape[0]

    L_single = window_length * (walk_length - window_length)
    L_all = walk_num * L_single
    pairs = -np.ones(L_all)

    #
    # Simulate the random walk
    #
    for sample_id in prange(walk_num):
        # Tie center and context nodes
        # Two node ids are converted to a single int id using the Canter pairng
        for t in range(L_single):
            t0, t1 = divmod(t, window_length)
            pid = sample_id * L_single + t
            t1 = t0 + t1 + 1
            if (walks[sample_id, t0] < 0) or (walks[sample_id, t1] < 0):
                continue
            # Cantor pairing function
            pairs[pid] = int(walks[sample_id, t0]) + int(walks[sample_id, t1])  #
            pairs[pid] = pairs[pid] * (pairs[pid] + 1) / 2 + int(walks[sample_id, t1])
    pairs = pairs[pairs >= 0]
    return pairs


def generate_center_context_pair(walks, window_length):
    return _generate_center_context_pair(walks.astype(np.int64), int(window_length))


@numba.jit(nopython=True, cache=True, parallel=True)
def _generate_center_context_pair(walks, window_length):
    """Generate center context node pairs from walks."""

    #
    # Allocate a memory for center and context node
    #
    walk_length = walks.shape[1]
    walk_num = walks.shape[0]

    L_single = window_length * (walk_length - window_length)
    L_all = walk_num * L_single
    pairs = -np.ones(L_all)

    #
    # Simulate the random walk
    #
    for sample_id in prange(walk_num):

        # Tie center and context nodes
        # Two node ids are converted to a single int id using the Canter pairng
        for t in range(L_single):
            t0, t1 = divmod(t, window_length)
            pid = sample_id * L_single + t
            t1 = t0 + t1 + 1
            if (walks[sample_id, t0] < 0) or (walks[sample_id, t1] < 0):
                continue
            # Cantor pairing function
            pairs[pid] = int(walks[sample_id, t0]) + int(walks[sample_id, t1])  #
            pairs[pid] = pairs[pid] * (pairs[pid] + 1) / 2 + int(walks[sample_id, t1])
    pairs = pairs[pairs >= 0]

    # Count center-context pairs
    freq = np.bincount(pairs.astype(np.int64))
    ids = np.nonzero(freq)[0]
    freq = freq[ids]

    # Deparing
    w = np.floor((np.sqrt(8 * ids + 1) - 1) * 0.5)
    t = (w ** 2 + w) * 0.5
    context = ids - t
    center = w - context
    return center, context, freq


#
# Compute the transition matrix
#
def calc_rwr(P, r, max_step, offset=0, w=None):
    if w is None:
        w = np.power(1 - r, np.arange(max_step))
        w = w / np.sum(w)

    Pt = sparse.csr_matrix(sparse.diags(np.ones(P.shape[0])))
    Ps = sparse.csr_matrix(sparse.diags(np.zeros(P.shape[0])))
    for i in range(max_step):
        Pt = P @ Pt
        if i < offset:
            continue
        Ps = Ps + w[i] * Pt
    return Ps


#
# Constructing a line graph
#
def construct_line_net_adj(A, p=1, q=1, add_source_node=True):
    """Construct the supra-adjacent matrix for the weighted network.

    The random walk process in the Node2Vec is the second order Markov process,
    where a random walker at node i remembers the previously visited node j
    and determines the next node k based on i and j.

    We transform the 2nd order Markov process to the 1st order Markov
    process on supra-nodes. Each supra-node represents a pair of
    nodes (j,i) in the original network, a pair of the previouly
    visited node and current node. We place an edge
    from a supra-node (j,i) to another supra-node (i,k) if the random walker
    can transit from node j, i and to k. The weight of edge is given by the
    unnormalized transition probability for the node2vec random walks.
    """

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    # Preprocessing
    # We efficiently simulate the random walk process using the scipy.csr_matrix
    # data structure. The followings are the preprocess for the simulation.
    A.sort_indices()  # the indices have to be sorted
    A_indptr_csr = A.indptr.astype(np.int32)
    A_indices_csr = A.indices.astype(np.int32)
    A_data_csr = A.data.astype(np.float32)
    A = A.T
    A.sort_indices()  # the indices have to be sorted
    A_indptr_csc = A.indptr.astype(np.int32)
    A_indices_csc = A.indices.astype(np.int32)
    A_data_csc = A.data.astype(np.float32)
    num_nodes = A.shape[0]

    # Make the edge list for the supra-networks
    supra_edge_list, edge_weight = construct_node2vec_supra_net_edge_pairs(
        A_indptr_csr,
        A_indices_csr,
        A_data_csr,
        A_indptr_csc,
        A_indices_csc,
        A_data_csc,
        num_nodes,
        p=p,
        q=q,
        add_source_node=add_source_node,
    )

    # Remove isolated nodes from the supra-network and
    # re-index the ids.
    supra_nodes, supra_edge_list = np.unique(supra_edge_list, return_inverse=True)
    supra_edge_list = supra_edge_list.reshape((edge_weight.size, 2))

    # Each row indicates the node pairs in the original net that constitute the supra node
    src_trg_pairs = (np.vstack(divmod(supra_nodes, num_nodes)).T).astype(int)

    # Construct the supra-adjacency matrix
    supra_node_num = supra_nodes.size
    Aspra = sparse.csr_matrix(
        (edge_weight, (supra_edge_list[:, 0], supra_edge_list[:, 1])),
        shape=(supra_node_num, supra_node_num),
    )
    return Aspra, src_trg_pairs


@numba.jit(nopython=True, cache=True)
def construct_node2vec_supra_net_edge_pairs(
    A_indptr_csr,
    A_indices_csr,
    A_data_csr,
    A_indptr_csc,
    A_indices_csc,
    A_data_csc,
    num_nodes,
    p,
    q,
    add_source_node,
):
    """Construct the weight and edges for the supra adjacency matrix for a
    network, where each supra-node represents a pair of source and target nodes
    in the original network. The supra-edge represents the edge weight for the
    node2vec biased random walk process.

    Parameters
    ----------
    A_indptr_csr : numpy.ndarray
        A_indptr_csr is given by A.indptr, where A is scipy.sparse.csr_matrix
        representing the adjacency matrix for the original network
    A_indices_csr : numpy.ndarray
        A_indices_csr is given by A.indices, where A is scipy.sparse.csr_matrix
        as in A_indptr_csr
    A_data_csr : numpy.ndarray
        A_data_csr is given by A.data, where A is the scipy.sparse.csr_matrix
        as in A_indptr_csr
    A_indptr_csc : numpy.ndarray
        A_indptr_csc is given by A.indptr, where A is scipy.sparse.csc_matrix
        representing the adjacency matrix for the original network
    A_indices_csc : numpy.ndarray
        A_indices_csc is given by A.indices, where A is scipy.sparse.csc_matrix
        as in A_indptr_csc
    A_data_csc : numpy.ndarray
        A_data_csc is given by A.data, where A is the scipy.sparse.csc_matrix
        as in A_indptr_csc
    num_nodes : int
        Number of nodes in the original network
    p : float
        Parameter for the biased random walk. A smaller value encourages
        the random walker to return to the previously visited node.
    q : float
        Parameter for the biased random walk. A smaller value encourages the
        random walker to go away from the previously visited node.

    Return
    ------
    supra_edge_list : np.ndarray
        Edge list for the supra nodes. The id of the node is given by
        source * num_node + target, where source and target is the
        ID of the node in the original network.
    edge_weight_list : np.ndarray
        Edge weight for the edges.
    """

    num_edges = 0
    for i in range(num_nodes):
        outdeg = A_indptr_csr[i + 1] - A_indptr_csr[i]
        indeg = A_indptr_csc[i + 1] - A_indptr_csc[i]
        num_edges += (
            outdeg * indeg
        )  # number of paths of length 2 intermediated by node i
        if add_source_node:
            # edges emanating from the starting supra-node (i,i)
            num_edges += outdeg

    supra_edge_list = -np.ones((num_edges, 2))
    edge_weight_list = -np.zeros(num_edges)
    edge_id = 0
    for i in range(num_nodes):

        # neighbors for the outgoing edges
        # (i)->(next_node)
        for next_nei_id in range(A_indptr_csr[i], A_indptr_csr[i + 1]):
            next_node = A_indices_csr[next_nei_id]
            edge_w = A_data_csr[next_nei_id]

            # neighbors for the incoming edges
            # (prev_node)->(i)
            for prev_node in A_indices_csc[A_indptr_csc[i] : A_indptr_csc[i + 1]]:
                w = edge_w

                # If the next_node and prev_node are the same
                if next_node == prev_node:
                    w = w / p
                else:
                    # Check if next_node is a common neighbor for (prev_node)
                    # and (i) (prev_out_neighbor)<-(prev_node)->(i)
                    # ->(next_node)
                    is_common_neighbor = False
                    for prev_out_neighbor in A_indices_csr[
                        A_indptr_csr[prev_node] : A_indptr_csr[prev_node + 1]
                    ]:
                        # If True, next_node is not the common neighbor
                        # because prev_out_neighbor is increasing.
                        if next_node < prev_out_neighbor:
                            break
                        if prev_out_neighbor == next_node:  # common neighbor
                            is_common_neighbor = True
                            break
                    if is_common_neighbor is False:
                        w = w / q

                # Add an edge between two supra-nodes, (prev_node, i)
                # and (i, next_node)  weighted by w. The ID of the
                # surpra-node composed of (u,v) is given by
                # u * num_nodes + v
                supra_edge_list[edge_id, 0] = prev_node * num_nodes + i
                supra_edge_list[edge_id, 1] = i * num_nodes + next_node
                edge_weight_list[edge_id] = w
                edge_id += 1

            if add_source_node:
                # Add a node
                # supra-node
                # commposed of
                # (i,i) which
                # othe random walker
                # starts from.
                supra_edge_list[edge_id, 0] = i * num_nodes + i
                supra_edge_list[edge_id, 1] = i * num_nodes + next_node
                edge_weight_list[edge_id] = edge_w
                edge_id += 1

    return supra_edge_list, edge_weight_list


def to_trans_mat(mat):
    """Normalize a sparse CSR matrix row-wise (each row sums to 1) If a row is
    all 0's, it remains all 0's.

    Parameters
    ----------
    mat : scipy.sparse.csr matrix
        Matrix in CSR sparse format
    Returns
    -------
    out : scipy.sparse.csr matrix
        Normalized matrix in CSR sparse format
    """
    denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
    return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat


def pairing(k1, k2, unordered=False):
    """Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function."""
    k12 = k1 + k2
    if unordered:
        return (k12 * (k12 + 1)) * 0.5 + np.minimum(k1, k2)
    else:
        return (k12 * (k12 + 1)) * 0.5 + k2


def depairing(z):
    """Inverse of Cantor pairing function http://en.wikipedia.org/wiki/Pairing_
    function#Inverting_the_Cantor_pairing_function."""
    w = np.floor((np.sqrt(8 * z + 1) - 1) * 0.5)
    t = (w ** 2 + w) * 0.5
    y = np.round(z - t).astype(np.int64)
    x = np.round(w - y).astype(np.int64)
    return x, y


#
# Generate sentences from walk sequence as the input for gensim
#
def walk2gensim_sentence(walks, window_length):
    sentences = []
    for i in range(walks.shape[0]):
        w = walks[i, :]
        w = w[(~np.isnan(w)) * (w >= 0)]
        sentences += [w.astype(str).tolist()]
    return sentences


#
# Randomized SVD
#
def rSVD(X, dim, **params):
    if isinstance(X, list):
        return _rSVD_submatrices(X, r=dim, **params)
    else:
        return _rSVD_matrix(X, r=dim, **params)


def _rSVD_matrix(X, r, p=10, q=1):
    """Randomized SVD.

    Parameters
    ----------
    X : scipy.csr_sparse_matrix
        Matrix to decompose
    r : int
        Rank of decomposed matrix
    p : int (Optional; Default p = 5)
        Oversampling
    q : int (Optional; Default q = 1)
        Number of power iterations

    Return
    ------
    U : numpy.ndrray
        Left singular vectors of size (X.shape[0], r)
    lams : numpy.ndarray
        Singular values of size (r,)
    V : numpy.ndarray
        Right singular vectors of size (X.shape[0], r)
    """
    Nr, Nc = X.shape
    dim = r + p
    R = np.random.randn(Nc, dim)
    Z = X @ R
    for _i in range(q):
        Z = X @ (X.T @ Z)
    Q, R = np.linalg.qr(Z, mode="reduced")
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    selected = np.argsort(np.abs(S))[::-1][0:r]
    return U[:, selected], S[selected], VT[selected, :]


def _rSVD_submatrices(mat_seq, r, p=10, q=1, fill_zero=1e-20):
    """Randomized SVD for decomposable matrix. We assume that the matrix is
    given by mat_seq[0] + mat_seq[1],...

    Parameters
    ----------
    mat_seq: list
        List of decomposed matrices
    r : int
        Rank of decomposed matrix
    p : int (Optional; Default p = 10)
        Oversampling
    q : int (Optional; Default q = 1)
        Number of power iterations
    fill_zero: float
        Replace the zero values in the transition matrix with this value.

    Return
    ------
    U : numpy.ndrray
        Left singular vectors of size (X.shape[0], r)
    lams : numpy.ndarray
        Singular values of size (r,)
    V : numpy.ndarray
        Right singular vectors of size (X.shape[0], r)
    """
    Nc = mat_seq[-1][-1].shape[1]
    dim = r + p

    R = np.random.randn(Nc, dim)  # Random gaussian matrix
    Z = mat_prod_matrix_seq(mat_seq, R)

    for _i in range(q):  # Power iterations
        zz = mat_prod_matrix_seq(Z.T, mat_seq)
        Z = mat_prod_matrix_seq(mat_seq, zz.T)
    Q, R = np.linalg.qr(Z, mode="reduced")

    Y = mat_prod_matrix_seq(Q.T, mat_seq)

    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    selected = np.argsort(np.abs(S))[::-1][0:r]

    U, S, VT = U[:, selected], S[selected], VT[selected, :]
    if isinstance(U, np.matrix):
        U = np.asarray(U)
    if isinstance(VT, np.matrix):
        VT = np.asarray(VT)
    if isinstance(S, np.matrix):
        S = np.asarray(S).reshape(-1)
    return U, S, VT


def asemble_matrix_from_list(matrix_seq):
    S = None
    for k in range(len(matrix_seq)):
        R = matrix_seq[k][0]
        for i in range(1, len(matrix_seq[k])):
            R = R @ matrix_seq[k][i]

        if sparse.issparse(R):
            R = R.toarray()

        if S is None:
            S = R
        else:
            S += R
    return S


def mat_prod_matrix_seq(A, B):
    def right_mat_prod_matrix_seq(A, matrix_seq):

        S = None
        for k in range(len(matrix_seq)):
            R = A @ matrix_seq[k][0]
            if sparse.issparse(R):
                R = R.toarray()

            for rid in range(1, len(matrix_seq[k])):
                R = R @ matrix_seq[k][rid]

            if S is None:
                S = R
            else:
                S = S + R
        return S

    def left_mat_prod_matrix_seq(matrix_seq, A):

        S = None
        for k in range(len(matrix_seq)):
            R = matrix_seq[k][-1] @ A
            if sparse.issparse(R):
                R = R.toarray()
            for rid in range(1, len(matrix_seq[k])):
                R = matrix_seq[k][-rid - 1] @ R

            if S is None:
                S = R
            else:
                S = S + R
        return S

    if isinstance(A, list) and not isinstance(B, list):
        return left_mat_prod_matrix_seq(A, B)
    elif isinstance(B, list) and not isinstance(A, list):
        return right_mat_prod_matrix_seq(A, B)


def mat_seq_transpose(matrix_seq):
    retval = []
    for _i, seq in enumerate(matrix_seq):
        _seq = []
        for _j, s in enumerate(seq[::-1]):
            _seq += [s.T]
        retval += [_seq]
    return retval


#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)


#
# Logarithm
#
def safe_log(x, minval=1e-6):
    return np.log(np.maximum(x, minval))
