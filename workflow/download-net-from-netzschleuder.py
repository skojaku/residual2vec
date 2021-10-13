"""Download the network from Netzschleuder:

https://networks.skewed.de
"""
import sys

import graph_tool.all as gt
import numpy as np

# %%
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

if "snakemake" in sys.modules:
    net_name = snakemake.params["net_name"]
    output_file = snakemake.output["output_file"]
else:
    net_name = "dblp_cite"
    output_file = "../data/"

#
# Load
#
g = gt.collection.ns[net_name]
A = gt.adjacency(g).T

#
# Get the largerst component
#
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

# %%
