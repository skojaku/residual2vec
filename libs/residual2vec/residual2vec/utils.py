"""Supplementary functions for residual2vec.py."""
import numpy as np
from numba import njit
from scipy import sparse


#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    """Convert to the adjacency matrix in form of sparse.csr_matrix.

    :param net: adjacency matrix
    :type net: np.ndarray or csr_matrix
    :return: adjacency matrix
    :rtype: sparse.csr_matrix
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)
    else:
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))


def row_normalize(mat):
    """Normalize the matrix row-wise.

    :param mat: matrix
    :type mat: sparse.csr_matrix
    :return: row normalized matrix
    :rtype: sparse.csr_matrix
    """
    denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
    return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat


def to_member_matrix(group_ids, node_ids=None, shape=None):
    """Create the binary member matrix U such that U[i,k] = 1 if i belongs to group k otherwise U[i,k]=0.

    :param group_ids: group membership of nodes. group_ids[i] indicates the ID (integer) of the group to which i belongs.
    :type group_ids: np.ndarray
    :param node_ids: IDs of the node. If not given, the node IDs are the index of `group_ids`, defaults to None.
    :type node_ids: np.ndarray, optional
    :param shape: Shape of the member matrix. If not given, (len(group_ids), max(group_ids) + 1), defaults to None
    :type shape: tuple, optional
    :return: Membership matrix
    :rtype: sparse.csr_matrix
    """
    if node_ids is None:
        node_ids = np.arange(len(group_ids))

    if shape is not None:
        Nr = int(np.max(node_ids) + 1)
        Nc = int(np.max(group_ids) + 1)
        shape = (Nr, Nc)
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (node_ids, group_ids)), shape=shape,
    )
    U.data = U.data * 0 + 1
    return U


def matrix_sum_power(A, T):
    """Take the sum of the powers of a matrix, i.e.,

    sum_{t=1} ^T A^t.

    :param A: Matrix to be powered
    :type A: np.ndarray
    :param T: Maximum order for the matrixpower
    :type T: int
    :return: Powered matrix
    :rtype: np.ndarray
    """
    At = np.eye(A.shape[0])
    As = np.zeros((A.shape[0], A.shape[0]))
    for _ in range(T):
        At = A @ At
        As += At
    return As


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


def safe_log(A, minval=1e-12):
    if sparse.issparse(A):
        A.data = np.log(np.maximum(A.data, minval))
        return A
    else:
        return np.log(np.maximum(A, minval))


@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


def csr_sampling(rows, csr_mat):
    return _csr_sampling(rows, csr_mat.indptr, csr_mat.indices, csr_mat.data)


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _csr_sampling(rows, indptr, indices, data):
    n = len(rows)
    retval = np.empty(n, dtype=indices.dtype)
    for j in range(n):
        neighbors = _neighbors(indptr, indices, rows[j])
        neighbors_p = _neighbors(indptr, data, rows[j])
        retval[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
    return retval
