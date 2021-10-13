import os

import graph_embeddings
import numpy as np
import pandas as pd
from scipy import sparse

import residual2vec

#
# Input
#
netfile = snakemake.input["netfile"]
nodefile = snakemake.input["nodefile"] if "nodefile" in snakemake.input.keys() else None
dim = int(snakemake.params["dim"])
window_length = int(snakemake.params["window_length"])
model_name = snakemake.params["model_name"]
directed = snakemake.params["directed"] == "directed"
noselfloop = (
    snakemake.params["noselfloop"] == "True"
    if "noselfloop" in snakemake.params.keys()
    else False
)
controll_for = (
    snakemake.params["controlfor"]
    if "controlfor" in snakemake.params.keys()
    else "None"
)
num_walks = (
    int(snakemake.params["num_walks"]) if "num_walks" in snakemake.params.keys() else 1
)
backward_prob = (
    float(snakemake.params["backward_prob"])
    if "backward_prob" in snakemake.params.keys()
    else 0
)
embfile = snakemake.output["embfile"]

net = sparse.load_npz(netfile)

if nodefile is not None:
    node_table = pd.read_csv(nodefile)

if directed is False:
    net = net + net.T

if noselfloop:
    net.setdiag(0)
    logger.debug("Remove selfloops")

if directed and model_name in ["residual2vec", "residual2vec-unbiased"]:
    eta = backward_prob / (1 - backward_prob)
    outdeg = np.array(net.sum(axis=1)).reshape(-1)
    indeg = np.array(net.sum(axis=0)).reshape(-1)
    eta_nodes = (
        outdeg * backward_prob / (indeg * (1 - backward_prob) + outdeg * backward_prob)
    )
    eta_nodes[outdeg == 0] = 1
    eta_nodes[indeg == 0] = 0
    net = sparse.diags(1 - eta_nodes) * net + sparse.diags(eta_nodes) @ net.T

#
# Embedding models
#
membership = np.zeros(net.shape[0])
offset = np.zeros(net.shape[0])
if model_name == "levy-word2vec":
    model = r2v.LevyWord2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "node2vec":
    model = r2v.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "node2vec-qhalf":
    model = r2v.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks, q=0.5
    )
elif model_name == "node2vec-qdouble":
    model = r2v.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks, q=2
    )
elif model_name == "node2vec-unbiased":
    model = r2v.Node2Vec(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
    model.w2vparams["ns_exponent"] = 0.0
elif model_name == "deepwalk":
    model = r2v.DeepWalk(
        window_length=window_length, restart_prob=0, num_walks=num_walks
    )
elif model_name == "glove":
    model = r2v.Glove(window_length=window_length, restart_prob=0, num_walks=num_walks)
elif model_name == "residual2vec-unbiased":
    model = r2v.Residual2Vec(
        null_model="erdos",
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="pairwise",
    )
elif model_name == "residual2vec-degcorr":
    gnum = 30
    net = sparse.csr_matrix(net)
    deg = np.array(net.sum(axis=0)).reshape(-1)
    rk = np.argsort(np.argsort(deg))
    membership = np.floor(rk / len(rk) * gnum).astype(int)
    model = r2v.Residual2Vec(
        null_model="constrained-configuration",
        group_membership=membership,
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="pairwise",
    )
elif model_name == "residual2vec-sim":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2VecSimple(
            num_walks=num_walks,
            null_model="configuration",
            window_length=window_length,
            restart_prob=0,
        )
    else:
        membership = node_table[controll_for].values
        model = r2v.Residual2VecSimple(
            num_walks=num_walks,
            null_model="constrained-configuration",
            group_membership=membership,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
        )
elif model_name == "residual2vec-truncated":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2VecTruncated(window_length=window_length, restart_prob=0,)
    else:
        membership = node_table[controll_for].values
        model = r2v.Residual2VecTruncated(
            group_membership=membership, window_length=window_length, restart_prob=0,
        )
elif model_name == "residual2vec-adap":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2VecAdaptive(
            num_walks=num_walks,
            null_model="configuration",
            window_length=window_length,
            restart_prob=0,
        )
    else:
        membership = node_table[controll_for].values
        model = r2v.Residual2VecAdaptive(
            num_walks=num_walks,
            null_model="constrained-configuration",
            group_membership=membership,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
        )
elif model_name == "fairwalk":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Fairwalk(window_length=window_length)
    else:
        membership = node_table[controll_for].values
        model = r2v.Fairwalk(group_membership=membership, window_length=window_length,)
elif model_name == "residual2vec":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2Vec(
            null_model="configuration",
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
        )
    else:
        membership = node_table[controll_for].values
        model = r2v.Residual2Vec(
            null_model="constrained-configuration",
            group_membership=membership,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
        )
elif model_name == "jresidual2vec-unbiased":
    model = r2v.Residual2Vec(
        null_model="erdos",
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="pairwise",
        train_by_joint_prob=True,
    )
elif model_name == "jresidual2vec":
    if (controll_for == "None") or (node_table is None):
        model = r2v.Residual2Vec(
            null_model="configuration",
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
            train_by_joint_prob=True,
        )
    else:
        membership = node_table[controll_for].values
        model = r2v.Residual2Vec(
            null_model="constrained-configuration",
            group_membership=membership,
            window_length=window_length,
            restart_prob=0,
            # num_walks=num_walks,
            residual_type="pairwise",
            train_by_joint_prob=True,
        )
elif model_name == "iresidual2vec-unbiased":
    model = r2v.Residual2Vec(
        null_model="erdos",
        window_length=window_length,
        restart_prob=0,
        # num_walks=num_walks,
        residual_type="individual",
    )
elif model_name == "leigenmap":
    model = r2v.LaplacianEigenMap()
elif model_name == "glee":
    model = r2v.GeometricLaplacian()
elif model_name == "netmf":
    model = r2v.NetMF(window_length=window_length)
elif model_name == "graphsage":
    model = r2v.GraphSage()
elif model_name == "gcn":
    model = r2v.GCN()
elif model_name == "graphsage-doubleK":
    model = r2v.GraphSage(num_default_features=dim * 2)
elif model_name == "gcn-doubleK":
    model = r2v.GCN(num_default_features=dim * 2)
elif model_name == "gat":
    model = r2v.GAT(layer_sizes=[64, 256])
elif model_name == "gat-doubleK":
    model = r2v.GCN(num_default_features=dim * 2)
elif model_name == "lndeg":  # fake embedding. Just to save offset
    A = sparse.csr_matrix(net)
    deg = np.array(A.sum(axis=1)).reshape(-1)
    emb = np.zeros((len(deg), dim))
    np.savez(
        embfile,
        emb=emb,
        out_emb=emb,
        membership=np.zeros_like(deg),
        offset=np.log(np.maximum(deg, 1)),
        window_length=window_length,
        dim=dim,
        directed=directed,
        model_name=model_name,
    )
    sys.exit()


#
# Embedding
#
model.fit(sparse.csr_matrix(net))
emb = model.transform(dim=dim)
out_emb = model.transform(dim=dim, return_out_vector=True)

try:
    offset = model.node_offset
except AttributeError:
    pass

#
# Save
#
np.savez(
    embfile,
    emb=emb,
    out_emb=out_emb,
    membership=membership,
    offset=offset,
    window_length=window_length,
    dim=dim,
    directed=directed,
    model_name=model_name,
)
