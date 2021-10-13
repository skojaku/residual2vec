# %%
# %load_ext autoreload
# %autoreload 2

import copy
import glob
import json
import os
import sys

# Plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rc("text", usetex=True)

# %% [markdown]
# parameterize

# %% tags=["parameters"]
res_file = "../data/link-prediction/results/auc_score.csv"
output_file = "../figs/result-link-prediction-gcn-graphsage.pdf"

# %% [markdown]
# Data

# %%
result_table = pd.read_csv(res_file)

# Rename data and method for plotting

# %%
data_labels = {
    # "polblogs": "Blog",
    "openflights": "Airport",
    "wiki-vote": "Wikipedia vote",
    "ppi": "Protein-Protein",
    "ca-AstroPh": "Coauthorship (AstroPh)",
    "ca-HepTh": "Coauthorship (HepTh)",
    "dblp": "Citation (DBLP)",
}
result_table["netName"] = result_table["net"].map(data_labels)


# %%
print(result_table["model"].drop_duplicates())
method_labels = {
    "gcn": "GCN ($\hat K = K$)",
    "gcn-doubleK": "GCN ($\hat K = 2K$)",
    "graphsage": "GraphSAGE ($\hat K= K$)",
    "graphsage-doubleK": "GraphSAGE ($\hat K= 2K$)",
    "gat": "GAT ($\hat K= K$)",
    "gat-doubleK": "GAT ($\hat K= 2K$)",
}
result_table["modelName"] = result_table["model"].map(method_labels)
# %% [markdown]

# %%

plot_kwargs = {"err_style": "band", "markersize": 8}

# Model specific keywords
palette = sns.color_palette().as_hex()

# To homogenize the space between x-ticks
result_table["dim"] = np.log(result_table["dim"]) / np.log(2)

# %%
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")
sns.set_color_codes("deep")

col_order = data_labels.values()

cmap = sns.color_palette().as_hex()
cmap_muted = sns.color_palette("pastel", desat=0.5).as_hex()
cmap = sum(
    [
        [cmap[0]],
        [cmap_muted[0]],
        [cmap[1]],
        [cmap_muted[1]],
        [cmap[2]],
        [cmap_muted[2]],
    ],
    [],
)

g = sns.FacetGrid(
    data=result_table,
    col="netName",
    col_wrap=3,
    col_order=col_order,
    height=3,
    aspect=1,
    sharex=True,
    sharey=True,
    hue="modelName",
    palette=cmap[: len(method_labels.values())],
    hue_order=list(method_labels.values()),
    hue_kws={
        "marker": ["o", "o", "s", "s", "d", "d"],
        "ls": ["-", "--", "-", "--", "-", "--"],
        "color": cmap,
        "markersize": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        "markerfacecolor": cmap,
        "markeredgecolor": ["k", "k", "k", "k", "k", "k", "k", "k", "k", "k", "k", "k"],
    },
    gridspec_kws={"wspace": 0},
)


g.map(sns.lineplot, "dim", "score", ci="sd")
# axins = inset_axes(g.axes.flat[0], 3, 0.5, loc="lower left", borderpad=3)
for ax in g.axes:
    ax.axhline(0.5, ls="--", color="grey", alpha=0.8)

g.set(
    xticks=np.arange(3, 8),
    xticklabels=[r"$2^{}$".format(i) for i in range(3, 8)],
    yticks=np.linspace(0.2, 1, 5),
    yticklabels=["%.1f" % d if d > 0 else "0" for d in np.linspace(0.2, 1, 5)],
    ylim=(0.4, 1.05),
    # ylim=(0.79, 1),
)

ax = g.axes[1]
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
    columnspacing=1,
    fontsize=15,
)
g.set_titles(col_template="{col_name}")

g.set_xlabels("")
g.set_ylabels("")

subcap = "ABCDEFGHIJK"
for i, ax in enumerate(g.axes):
    ax.annotate(
        r"{\bf %s}" % subcap[i],
        (-0.05, 1.01),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=16,
    )

plt.subplots_adjust(wspace=0.1)
fig = plt.gcf()
fig.text(0.03, 0.5, "AUC-ROC", rotation=90, ha="center", va="center", fontsize=18)
fig.text(0.5, 0.04, "Dimension, $K$", ha="center", va="center", fontsize=18)
fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%

# %%
