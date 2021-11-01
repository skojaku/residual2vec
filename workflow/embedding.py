import os
import sys

sys.path.append(os.path.abspath(os.path.join("./libs/graph_embeddings")))
import graph_embeddings
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.append(os.path.abspath(os.path.join("./libs/residual2vec")))
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
backward_prob = (
    float(snakemake.params["backward_prob"])
    if "backward_prob" in snakemake.params.keys()
    else 0
)
embfile = snakemake.output["embfile"]

#
# Load
#
net = sparse.load_npz(netfile)

if nodefile is not None:
    node_table = pd.read_csv(nodefile)

#
# Preprocess
#
if directed is False:
    net = net + net.T

if noselfloop:
    net.setdiag(0)

if directed:
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
# Load the emebdding models
#
membership = np.zeros(net.shape[0])
offset = np.zeros(net.shape[0])
if model_name == "node2vec":
    model = graph_embeddings.Node2Vec(
        window_length=window_length, restart_prob=0
    )
elif model_name == "node2vec-qhalf":
    model = graph_embeddings.Node2Vec(
        window_length=window_length, restart_prob=0, q=0.5
    )
elif model_name == "node2vec-qdouble":
    model = graph_embeddings.Node2Vec(
        window_length=window_length, restart_prob=0, q=2
    )
elif model_name == "deepwalk":
    model = graph_embeddings.DeepWalk(
        window_length=window_length, restart_prob=0, 
    )
elif model_name == "glove":
    model = graph_embeddings.Glove(
        window_length=window_length, restart_prob=0,
    )
elif model_name == "fairwalk":
    if (controll_for == "None") or (node_table is None):
        model = graph_embeddings.Fairwalk(window_length=window_length)
    else:
        membership = node_table[controll_for].values
        model = graph_embeddings.Fairwalk(
            group_membership=membership, window_length=window_length,
        )
elif model_name == "residual2vec":
    if (controll_for == "None") or (node_table is None):
        model = residual2vec.residual2vec(window_length=window_length,)
    else:
        membership = node_table[controll_for].values
        model = residual2vec.residual2vec(
            group_membership=membership, window_length=window_length,
        )
elif model_name == "leigenmap":
    model = graph_embeddings.LaplacianEigenMap()
elif model_name == "netmf":
    model = graph_embeddings.NetMF(window_length=window_length)
elif model_name == "graphsage":
    model = graph_embeddings.GraphSage()
elif model_name == "gcn":
    model = graph_embeddings.GCN()
elif model_name == "graphsage-doubleK":
    model = graph_embeddings.GraphSage(num_default_features=dim * 2)
elif model_name == "gcn-doubleK":
    model = graph_embeddings.GCN(num_default_features=dim * 2)
elif model_name == "gat":
    model = graph_embeddings.GAT(layer_sizes=[64, 256])
elif model_name == "gat-doubleK":
    model = graph_embeddings.GCN(num_default_features=dim * 2)
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
    membership=membership,
    offset=offset,
    window_length=window_length,
    dim=dim,
    directed=directed,
    model_name=model_name,
)
