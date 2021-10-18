# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
#
# Input
#
try:
    result_file = snakemake.input["result_file"]
    json_file = snakemake.input["config_file"]
    output_file = snakemake.output["output_file"]
except NameError:
    result_file = "../data/lfr-benchmark/results/results.csv"
    json_file = "../data/lfr-benchmark/lfr-config.json"
    output_file = "../figs/lfr-result-gcn-graphsage.pdf"

# %%
#
# Load
#
result_table = pd.read_csv(result_file)
with open(json_file, "r") as f:
    config_file = json.load(f)
config_table = pd.DataFrame(config_file)
config_table["param_id"] = np.arange(config_table.shape[0])
result_table = pd.merge(result_table, config_table, on="param_id", how="left")

# %%
#
# Slice data used for plotting
#
# dis_metric = "euclidean"
dis_metric = "cosine"
eval_metrics = ["auc"]
# metrics = ["nmi", "esim", "f1score"]
usecols = ["s", "mu", "param_id", "model", "dim", "tau", "N", "metric"]
dim = 64
# param_ids = [4, 5]
# param_ids = [6, 7]
param_ids = [0, 1]
# param_ids = [2, 3]

dflist = []
for m in eval_metrics:
    df = result_table[usecols + [m]].copy().rename(columns={m: "score"})
    df["scoreType"] = m
    dflist += [df]
result_table = pd.concat(dflist, ignore_index=True)
result_table = result_table[result_table.dim == dim]
result_table = result_table[result_table.metric == dis_metric]
result_table = result_table[result_table.param_id.isin(param_ids)]
# %%

# %%
#
# Rename
#
method_labels = {
    "gcn": "GCN ($\hat K = K$)",
    "gcn-doubleK": "GCN ($\hat K = 2K$)",
    "graphsage": "GraphSAGE ($\hat K= K$)",
    "graphsage-doubleK": "GraphSAGE ($\hat K= 2K$)",
    "gat": "GAT ($\hat K= K$)",
    "gat-doubleK": "GAT ($\hat K= 2K$)",
}
result_table["modelName"] = result_table["model"].map(method_labels)
# %%
#
# Color
#
sns.set(font_scale=1.2)
sns.set_style("ticks")
sns.set_style("whitegrid")
cmap = sns.color_palette()


#
# Plot
#

model_order = method_labels.values()
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
    row="tau",
    col="scoreType",
    hue="modelName",
    height=3.5,
    aspect=1.2,
    palette=cmap[: len(method_labels.values())],
    hue_order=list(method_labels.values()),
    row_order=[6, 2],
    hue_kws={
        "marker": ["o", "o", "s", "s", "d", "d"],
        "ls": ["-", "--", "-", "--", "-", "--"],
        "color": cmap,
        "markersize": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        "markerfacecolor": cmap,
        "markeredgecolor": ["k", "k", "k", "k", "k", "k", "k", "k", "k", "k", "k", "k"],
    },
)

g.map(sns.lineplot, "mu", "score", ci="sd")
# g.map(sns.lineplot, "mu", "score", err_style="band")

ax = g.axes.flat[1]
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(-0.2, -0.3),
    frameon=False,
    handletextpad=0.15,
    columnspacing=0.5,
    ncol=2,
)

g.set(xlim=(0, 1))
g.set(ylim=(0.5, 1.02))
g.axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
# g.set(xlim=(0, 1))
# g.set(ylim=(0, 1.05))
# g.set_titles("")
g.set_xlabels("")
g.set_ylabels("")


# subcap = "GH"
# subcap = "GHIJK"
# for i, ax in enumerate(g.axes.flat):
#    ax.annotate(
#        r"%s" % subcap[i],
#        (-0.1, 1.04),
#        xycoords="axes fraction",
#        ha="left",
#        va="bottom",
#        fontsize=18,
#        weight="bold",
#    )
t0 = g.axes.flat[0].set_title("Degree homogeneous", fontsize=14)
g.axes.flat[1].set_title("Degree heterogeneous", fontsize=14)

fig = plt.gcf()

t1 = fig.text(
    0.04, 0.5, r"AUC-ROC", ha="center", va="center", fontsize=18, rotation=90,
)
t2 = fig.text(
    0.55, 0.03, r"Mixing parameter, $\mu$", ha="center", va="center", fontsize=18
)
g.fig.savefig(output_file, dpi=300, bbox_extra_artists=[lgd, t1], bbox_inches="tight")
# fig.savefig(output_file, dpi=300, bbox_anchor="tight")

# %%

# %%
