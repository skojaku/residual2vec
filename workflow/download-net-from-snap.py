"""Download the network from SNAP:

https://snap.stanford.edu/data/
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

if "snakemake" in sys.modules:
    url = snakemake.params["url"]
    output_file = snakemake.output["output_file"]
else:
    url = ""
    output_file = "../data/"

#
# Load
#
df = pd.read_csv(
    url,
    compression="gzip",
    comment="#",
    sep="\t",
    names=["src", "trg"],
    dtype={"stc": int, "trg": int},
)

#
# Re-index
#
edges = df[["src", "trg"]].values.reshape(-1)
edges = np.unique(edges, return_inverse=True)[1]
edges = edges.reshape((-1, 2))

#
# Get the largerst component
#
N = int(np.max(edges.reshape(-1)) + 1)
A = sparse.csr_matrix(
    (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N)
)

_components, labels = connected_components(
    csgraph=A, directed=False, return_labels=True
)
lab, sz = np.unique(labels, return_counts=True)
inLargestComp = np.where(lab[np.argmax(sz)] == labels)[0]
A = A[inLargestComp, :][:, inLargestComp]

#
# Save
#
r, c, _ = sparse.find(A)
df = pd.DataFrame({"src": r, "trg": c})
df.to_csv(output_file, index=False, header=False)
