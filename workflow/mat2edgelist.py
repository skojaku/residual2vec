import pandas as pd
import scipy.io
from scipy import sparse
import numpy as np
from scipy.sparse.csgraph import connected_components

input_file = snakemake.input["input_file"]
output_file = snakemake.output["output_file"]

A = scipy.io.loadmat(input_file)["network"]
_components, labels = connected_components(
    csgraph=A, directed=False, return_labels=True
)
lab, sz = np.unique(labels, return_counts=True)
inLargestComp = np.where(lab[np.argmax(sz)] == labels)[0]
A = A[inLargestComp, :][:, inLargestComp]


src, trg, _ = sparse.find(A)
pd.DataFrame({"src": src, "trg": trg}).to_csv(output_file, index=False, header=False)
