import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

# Science genome lib
sys.path.append(os.path.abspath(os.path.join("./libs/link_prediction")))
from link_prediction.network_train_test_splitter import NetworkTrainTestSplitterWithMST
from scipy import sparse

#
# Input
#
edge_file = snakemake.input["edge_file"]
net_file = snakemake.output["net_file"]
test_edge_file = snakemake.output["test_edge_file"]
removal_fraction = float(snakemake.params["removal_frac"])
directed = snakemake.params["directed"] == "directed"

# Construct a graph
df = pd.read_csv(edge_file, header=None)
nodes, edges = np.unique(df.values.reshape(-1), return_inverse=True)
edges = edges.reshape((df.shape[0], 2))
N = len(nodes)
A = sparse.csr_matrix(
    (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N)
)
G = nx.from_scipy_sparse_matrix(A)

# Test and train edge split
splitter = NetworkTrainTestSplitterWithMST(
    G, directed=directed, fraction=removal_fraction
)
splitter.train_test_split()
splitter.generate_negative_edges()


#
# Retrieve results
#
Ab = nx.adjacency_matrix(splitter.G)
df = pd.DataFrame(splitter.test_edges)
df["edge_type"] = 1
dg = pd.DataFrame(splitter.negative_edges)
dg["edge_type"] = -1
test_edges = pd.concat([df, dg], ignore_index=True)

#
# Save results
#
sparse.save_npz(net_file, Ab)
test_edges.to_csv(test_edge_file, index=True)
