import sys

import graph_tool.all as gt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from residual_node2vec import utils
from scipy import sparse
from sklearn import metrics


def find_blocks_by_sbm(A, K, directed=False):
    def scipy_to_graph_tool(adj, directed=False):
        g = gt.Graph(directed=directed)
        edge_weights = g.new_edge_property("double")
        g.edge_properties["weight"] = edge_weights
        r, c, v = sparse.find(adj)

        g.add_edge_list(
            np.vstack([r, c, v]).T, eprops=[edge_weights],
        )
        return g

    g = scipy_to_graph_tool(A, directed)
    state = gt.minimize_blockmodel_dl(g, deg_corr=True, B_min=K, B_max=K)
    return np.array(state.get_blocks().a)


def log_trans_prob(A, window_length, restart_prob=0):
    weights = np.power((1 - restart_prob), np.arange(window_length))
    weights = weights / np.sum(weights)
    P = utils.to_trans_mat(A)
    P = utils.calc_rwr(P, None, window_length, w=weights)
    P.data = utils.safe_log(P.data)
    return P


def approx_log_trans_prob(
    A, window_length, beta=0.66, restart_prob=0, approx_order=2, directed=False
):

    if window_length <= approx_order:
        logP = log_trans_prob(A, window_length=window_length, restart_prob=restart_prob)
        return [[logP]]

    K = np.ceil(np.power(A.shape[0], beta)).astype(int)
    cids = find_blocks_by_sbm(A, K, directed=directed)

    num_nodes = A.shape[0]
    num_coms = np.max(cids) + 1
    U = sparse.csr_matrix(
        (np.ones(num_nodes), (np.arange(num_nodes), cids)), shape=(num_nodes, num_coms)
    )

    # Compute the parameter for the SBM
    din = np.array(A.sum(axis=0)).reshape(-1)
    Din = np.array(din @ U).reshape(-1)
    theta_in = din / Din[cids]
    adj_com = U.T @ A @ U
    trans_com = utils.to_trans_mat(adj_com)

    #
    # Precompute the transition matrix for short steps
    #
    P = utils.to_trans_mat(A)
    Pt = [P]
    for t in range(1, approx_order):
        Pt += [P @ Pt[t - 1]]

    #
    # Calculate the trans prob for short steps
    #
    weights = np.power((1 - restart_prob), np.arange(window_length))
    weights = weights / np.sum(weights)
    Pshort = weights[0] * P.copy()
    for t in range(1, approx_order):
        Pshort += weights[t] * Pt[t]

    #
    # Approximate the long steps by SBM
    #

    # Compute the trans prob from node to a community
    n2c = utils.calc_rwr(
        trans_com, None, window_length - approx_order, w=weights[approx_order:]
    ).toarray()
    n2c = np.array((Pt[approx_order - 1] @ U) @ n2c)

    # Make a mask to prevent exp(0) = 1. 0 in n2 means 0
    mask = n2c.copy()
    mask[mask > 0] = 1

    log_theta_in = theta_in.copy()
    log_theta_in[theta_in > 0] = np.log(theta_in[theta_in > 0])
    ThetaOut = sparse.csr_matrix(U.T) @ sparse.diags(log_theta_in)

    logn2c = n2c.copy()
    logn2c[n2c > 0] = np.log(logn2c[n2c > 0])

    logP_long = [
        [logn2c, sparse.csr_matrix(U.T)],
        [mask, ThetaOut],
    ]

    #
    # Merge
    #
    r, c, v = sparse.find(Pshort)
    v = (
        np.log(v + n2c[(r, cids[c])] * theta_in[c])
        - logn2c[(r, cids[c])]
        - np.log(theta_in[c])
    )
    # utils.safe_log(
    #    n2c[(r, cids[c])] * theta_in[c]
    # )
    logP_short = sparse.csr_matrix(
        (np.array(v).reshape(-1), (r, c)), shape=(num_nodes, num_nodes)
    )
    logP = [[logP_short]] + logP_long
    return logP


if "snakemake" in sys.modules:
    edge_file = snakemake.input["edge_file"]
    output_file = snakemake.output["output_file"]
    approx_order = snakemake.params["approx_order"]
    sample_max = 20000
else:
    edge_file = "../../../data/link-prediction/data/polblogs.csv"
    output_file = "../../../data/log-trans-approximation/yypred-polblogs.csv"
    approx_order = 2
    sample_max = 20000

edge_table = pd.read_csv(edge_file, header=None, names=["src", "trg"]).astype(int)
N = np.max(edge_table.values) + 1
net = sparse.csr_matrix(
    (np.ones(edge_table.shape[0]), (edge_table["src"], edge_table["trg"])), shape=(N, N)
)


#
# Load networks
#
net = net + 0.01 * net.T  # to prevent the dangling nodes

#
# Calculate the log probability
#
dflist = []
for window_length in [3, 5, 10, 20]:
    logPapx = approx_log_trans_prob(
        net, window_length, approx_order=approx_order, directed=True
    )
    logPapx = utils.mat_prod_matrix_seq(logPapx, np.eye(net.shape[0]))
    logP = log_trans_prob(net, window_length)

    #
    # Formatting
    #
    y, ypred = logP.toarray().reshape(-1), logPapx.reshape(-1)

    s = ~np.isclose(y, 0)
    y, ypred = y[s], ypred[s]

    #
    # Save results
    #
    df = pd.DataFrame({"y": y, "ypred": ypred})
    df["window_length"] = window_length

    if df.shape[0] > sample_max:
        df = df.sample(sample_max)

    dflist += [df]
df = pd.concat(dflist, ignore_index=True)
df.to_csv(output_file)
# %%
