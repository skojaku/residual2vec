# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-04-19 20:22:52
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-13 10:47:03
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
A = sparse.csr_matrix(A + A.T)


#
#
# Embed
#
model = rv.residual2vec_sgd(
    window_length=5,
    num_walks=40,
    RP=True,
    noise_sampler=rv.ConfigModelNodeSampler(),
    # context_window_type="right",
    device="cuda:0",
)
emb = model.fit(A).transform(dim=64)

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
# sns.heatmap(emb @ emb.T, vmax=0.5, vmin=-0.5, cmap="coolwarm")
sns.heatmap(emb @ emb.T)

# %%
emb = rv.fastRP.fastRP(A, dim=100, window_size=3, normalize=True)
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
sns.heatmap(emb @ emb.T)
