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
    output_zoomed_file = snakemake.output["output_zoomed_file"]
except NameError:
    result_file = "../data/lfr-benchmark/results/results.csv"
    json_file = "../data/lfr-benchmark/lfr-config.json"
    output_file = "../figs/result-lfr.pdf"
    output_zoomed_file = "../figs/result-zoomed-lfr.pdf"

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
dis_metric = "cosine"
eval_metrics = ["auc"]
usecols = ["s", "mu", "param_id", "model", "dim", "tau", "N", "metric"]
dim = 64
param_ids = [0, 1]

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
    "residual2vec-truncated": "r2v-config",
    "node2vec": "node2vec ($q=1$)",
    "node2vec-qhalf": "node2vec ($q=0.5$)",
    "node2vec-qdouble": "node2vec ($q=2$)",
    "deepwalk": "DeepWalk",
    "glove": "Glove",
    "netmf": "NetMF",
    "leigenmap": "LEM",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
}
method_labels = {
    "residual2vec-truncated": "r2v-config",
    "node2vec": "node2vec (q=1)",
    "node2vec-qhalf": "node2vec ($q=0.5$)",
    "node2vec-qdouble": "node2vec ($q=2$)",
    "deepwalk": "DeepWalk",
    "fairwalk": "Fairwalk",
    "glove": "Glove",
    "netmf": "NetMF",
    "leigenmap": "LEM",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
}
result_table["modelName"] = result_table["model"].map(method_labels)
print(result_table["model"].drop_duplicates())

# %%
# Plot parameters
#
sns.set(font_scale=1.2)
sns.set_style("ticks")
sns.set_style("whitegrid")

plot_kwargs = {"err_style": "bars", "markersize": 10, "ci": 90}

model_order = method_labels.values()
# Color
cmap = sns.color_palette().as_hex()
cmap_muted = sns.color_palette("muted", desat=0.6).as_hex()
cmap_rw = sns.light_palette(
    sns.color_palette("Set2", desat=0.7).as_hex()[1], n_colors=6
)
cmap_rw2 = sns.light_palette(
    sns.color_palette("Set2", desat=0.7).as_hex()[4], n_colors=6
)
cmap_mf = sns.light_palette(
    sns.color_palette("Set2", desat=0.7).as_hex()[2], n_colors=6
)
cmap_gcn = sns.light_palette(
    sns.color_palette("Set2", desat=0.7).as_hex()[3], n_colors=6
)
model2color = {
    "residual2vec-truncated": "red",
    "node2vec": cmap_rw[5],
    "node2vec-qhalf": cmap_rw[3],
    "node2vec-qdouble": cmap_rw[1],
    "deepwalk": cmap_rw2[5],
    "fairwalk": cmap_rw2[3],
    "glove": cmap_mf[5],
    "netmf": cmap_mf[3],
    "leigenmap": cmap_mf[2],
    "gcn": cmap_gcn[5],
    "graphsage": cmap_gcn[4],
    "gat": cmap_gcn[3],
}
model2markeredgecolor = {
    "residual2vec-truncated": "k",
    "node2vec": "k",
    "node2vec-qhalf": "#8d8d8d",
    "node2vec-qdouble": "w",
    "deepwalk": "k",
    "fairwalk": "#8d8d8d",
    "glove": "k",
    "netmf": "w",
    "leigenmap": "#8d8d8d",
    "gcn": "k",
    "graphsage": "#8d8d8d",
    "gat": "k",
}
model2marker = {
    "residual2vec-truncated": "s",
    "node2vec": "s",
    "node2vec-qhalf": "o",
    "node2vec-qdouble": "D",
    "deepwalk": "s",
    "fairwalk": "o",
    "glove": "s",
    "netmf": "D",
    "leigenmap": "^",
    "gcn": "s",
    "graphsage": "o",
    "gat": "D",
}

model2ls = {  # line style
    "residual2vec-truncated": "-",
    "node2vec": "-",
    "node2vec-qhalf": ":",
    "node2vec-qdouble": "-.",
    "deepwalk": "-",
    "fairwalk": "-",
    "glove": "-",
    "netmf": "-",
    "leigenmap": "-",
    "gcn": "-",
    "graphsage": ":",
    "gat": "-.",
}

# hue order. Flip the order for legend
hue_order = list(method_labels.values())[::-1]
palette = [model2color[i] for i in method_labels.keys()][::-1]
markers = [model2marker[i] for i in method_labels.keys()][::-1]
markeredgecolors = [model2markeredgecolor[i] for i in method_labels.keys()][::-1]
linestyles = [model2ls[i] for i in method_labels.keys()][::-1]

# hue parameter kwd GridFacet
hue_kws = {
    "marker": markers,
    "ls": linestyles,
    "markersize": [6 for i in range(len(palette))],
    "markerfacecolor": palette,
    "color": palette,
    "markeredgecolor": markeredgecolors,
}

# %%
#
# Main plot
#
sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_style("ticks")


g = sns.FacetGrid(
    data=result_table,
    row="tau",
    col="scoreType",
    hue="modelName",
    height=3.5,
    aspect=1.2,
    palette=palette,
    hue_order=hue_order,
    row_order=[6, 2],
    hue_kws=hue_kws,
)

g.map(sns.lineplot, "mu", "score", **plot_kwargs)

g.set(xlim=(0.0, 1.0))
g.set(ylim=(0.5, 1.02))
g.axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.set_xlabels("")
g.set_ylabels("")
g.axes[1, 0].legend()
handles, labels = g.axes[1, 0].get_legend_handles_labels()
lgd = g.axes[1, 0].legend(
    handles[::-1],
    labels[::-1],
    loc="upper left",
    bbox_to_anchor=(-0.2, -0.3),
    frameon=False,
    handletextpad=0.15,
    columnspacing=0.5,
    ncol=3,
)
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

# %%

# %%
#
# Zoomed
#
g = sns.FacetGrid(
    data=result_table,
    row="tau",
    col="scoreType",
    hue="modelName",
    height=3.5,
    aspect=1.2,
    palette=palette,
    hue_order=hue_order,
    row_order=[6, 2],
    hue_kws=hue_kws,
)

g.map(sns.lineplot, "mu", "score", **plot_kwargs)

g.set(xlim=(0.39, 0.72))
g.set(ylim=(0.5, 1.02))
g.axes[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.axes[1, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.set_xlabels("")
g.set_ylabels("")
g.axes[1, 0].legend()
handles, labels = g.axes[1, 0].get_legend_handles_labels()
lgd = g.axes[1, 0].legend(
    handles[::-1],
    labels[::-1],
    loc="upper left",
    bbox_to_anchor=(-0.2, -0.3),
    frameon=False,
    handletextpad=0.15,
    columnspacing=0.5,
    ncol=3,
)


t0 = g.axes.flat[0].set_title("Degree homogeneous", fontsize=14)
g.axes.flat[1].set_title("Degree heterogeneous", fontsize=14)

fig = plt.gcf()

t1 = fig.text(
    0.04, 0.5, r"AUC-ROC", ha="center", va="center", fontsize=18, rotation=90,
)
t2 = fig.text(
    0.55, 0.03, r"Mixing parameter, $\mu$", ha="center", va="center", fontsize=18
)
sns.despine(right=False)
g.fig.savefig(
    output_zoomed_file, dpi=300, bbox_extra_artists=[lgd, t1], bbox_inches="tight"
)

# %%

# %%
#
# Calculate the performance grain
#

# Calculate the mean score
score_table = (
    result_table.groupby(["model", "dim", "tau", "metric", "mu"])
    .agg("mean")["score"]
    .reset_index()
)
score_table
# %%


# Split
ref_table = score_table[score_table["model"] == "residual2vec-truncated"].copy()
ref_table = ref_table.rename(columns={"score": "score_base"}).drop(columns="model")

df = pd.merge(score_table, ref_table, on=["dim", "tau", "metric", "mu"])
df["gain"] = np.round(100 * df["score"] / df["score_base"])

df = df[df.mu.between(0.3, 0.5)]

df.groupby(["model", "net"]).agg("mean")

# %%
