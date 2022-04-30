"""Test by visual inspection of the generated embeddings."""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA

import residual2vec as rv

#
# Load
#
edge_table = pd.read_csv("../data/edges.csv")

#
# Construct the graph
#
N = int(np.max(edge_table.max().values + 1))
A = sparse.csr_matrix(
    (np.ones(edge_table.shape[0]), (edge_table["src"], edge_table["trg"])), shape=(N, N)
)
A = sparse.csr_matrix(sparse.triu(A))


#
#
# Embed
#
model = rv.residual2vec_sgd(
    window_length=50,
    noise_sampler=rv.ConfigModelNodeSampler(),
    context_window_type="right",
)
emb = model.fit(A).transform(dim=5)

xy = PCA(n_components=2).fit_transform(emb)
n = int(N / 2)
group_ids = np.zeros(N)
group_ids[n:] = 1

#
# Plot the adjacency matrix
#
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

plt.imshow(A.toarray())


#
# Plot the PCA-projected embedding
#
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=group_ids)
ax.axis("off")
ax.legend().remove()

# %%
