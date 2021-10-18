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

# parameterize
if "snakemake" in sys.modules:
    res_file = snakemake.input["res_file"]
    output_file = snakemake.output["output_file"]
    output_zoomed_file = snakemake.output["output_file"]
else:
    res_file = "../data/link-prediction/results/auc_score.csv"
    output_file = "../figs/result-link-prediction.pdf"
    output_zoomed_file = "../figs/result-zoomed-link-prediction.pdf"
# %%
result_table = pd.read_csv(res_file)

# %%
result_table["model"].drop_duplicates()

# %%
# Data to plot
#
data_labels = {
    "openflights": "Airport",
    "wiki-vote": "Wikipedia vote",
    "ppi": "Protein-Protein",
    "ca-AstroPh": "Coauthorship (AstroPh)",
    "ca-HepTh": "Coauthorship (HepTh)",
    "dblp": "Citation (DBLP)",
}
result_table["netName"] = result_table["net"].map(data_labels)

method_labels = {
    "residual2vec": "r2v-config (w/ offset)",
    "residual2vec-dotsim": "r2v-config (w/o offset)",
    "node2vec": "node2vec (q=1)",
    "node2vec-qhalf": "node2vec ($q=0.5$)",
    "node2vec-qdouble": "node2vec ($q=2$)",
    "deepwalk": "DeepWalk",
    "fairwalk": "Fairwalk",
    "glove": "Glove (w/ offset)",
    "glove-dotsim": "Glove (w/o offset)",
    "netmf": "NetMF",
    "leigenmap": "LEM",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "lndeg": "Offset $z_i + z_j$ of\n residual2vec",
}
# To homogenize the space between x-ticks
result_table["modelName"] = result_table["model"].map(method_labels)
result_table["dim"] = np.log(result_table["dim"]) / np.log(2)

# %%
#
# Plot parameters
#

# Plot kwd GridFacet
plot_kwargs = {"err_style": "bars", "markersize": 6, "ci": 90}

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
    "residual2vec": "red",
    "residual2vec-dotsim": sns.light_palette("red", n_colors=5)[1],
    "node2vec": cmap_rw[5],
    "node2vec-qhalf": cmap_rw[3],
    "node2vec-qdouble": cmap_rw[1],
    "deepwalk": cmap_rw2[5],
    "fairwalk": cmap_rw2[3],
    "glove": cmap_mf[5],
    "glove-dotsim": cmap_mf[4],
    "netmf": cmap_mf[3],
    "leigenmap": cmap_mf[2],
    "gcn": cmap_gcn[5],
    "graphsage": cmap_gcn[4],
    "gat": cmap_gcn[3],
    "lndeg": "#afafaf",
}
model2markeredgecolor = {
    "residual2vec": "k",
    "residual2vec-dotsim": "#8d8d8d",
    "node2vec": "k",
    "node2vec-qhalf": "#8d8d8d",
    "node2vec-qdouble": "w",
    "deepwalk": "k",
    "fairwalk": "#8d8d8d",
    "glove": "k",
    "glove-dotsim": "#8d8d8d",
    "netmf": "w",
    "leigenmap": "#8d8d8d",
    "gcn": "k",
    "graphsage": "#8d8d8d",
    "gat": "w",
    "lndeg": "k",
}
model2marker = {
    "residual2vec": "s",
    "residual2vec-dotsim": "o",
    "node2vec": "s",
    "node2vec-qhalf": "o",
    "node2vec-qdouble": "D",
    "deepwalk": "s",
    "fairwalk": "o",
    "glove": "s",
    "glove-dotsim": "o",
    "netmf": "D",
    "leigenmap": "^",
    "gcn": "s",
    "graphsage": "o",
    "gat": "D",
    "lndeg": "s",
}

model2ls = {  # line style
    "residual2vec": "-",
    "residual2vec-dotsim": ":",
    "node2vec": "-",
    "node2vec-qhalf": ":",
    "node2vec-qdouble": "-.",
    "deepwalk": "-",
    "fairwalk": "-",
    "glove": "-",
    "glove-dotsim": ":",
    "netmf": "-",
    "leigenmap": "-",
    "gcn": "-",
    "graphsage": ":",
    "gat": "-.",
    "lndeg": ":",
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

# Model specific keywords
palette = sns.color_palette().as_hex()


#
#
# Main plot
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")
sns.set_color_codes("deep")

col_order = list(data_labels.values())

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
    palette=palette,
    hue_order=hue_order,
    hue_kws=hue_kws,
    gridspec_kws={"wspace": 0},
)


g.map(sns.lineplot, "dim", "score", **plot_kwargs)
# g.map(sns.lineplot, "dim", "score", ci="sd")
for ax in g.axes:
    ax.axhline(0.5, ls="--", color="grey", alpha=0.8)

g.set(
    xticks=np.arange(3, 8),
    xticklabels=[r"$2^{}$".format(i) for i in range(3, 8)],
    # yticks=np.linspace(0.2, 1, 5),
    # yticklabels=["%.1f" % d if d > 0 else "0" for d in np.linspace(0.75, 1, 5)],
    yticks=np.linspace(0.8, 1, 5),
    yticklabels=[
        "%.2f" % d if d not in [0, 1] else ["0", "1"][int(d)]
        for d in np.linspace(0.8, 1, 5)
    ],
    ylim=(0.79, 1.01),
)

ax = g.axes[3]
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(-0.18, -0.25),
    ncol=4,
    columnspacing=1,
    fontsize=13,
)

g.set_xlabels("")
g.set_ylabels("")

subcap = "ABCDEFGHIJK"
g.set_titles(col_template="")
for i, ax in enumerate(g.axes):
    # ax.set_title(
    #    r"\{\bf {subcap}\} {name}".format(subcap=subcap[i], name=col_order[i]),
    #    ha="left",
    # )
    ax.annotate(
        (r"{\bf %s}" % subcap[i]) + r" {name}".format(name=col_order[i]),
        (0.05, 0.92),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=14,
    )
# ax.title.set_position([0.5, 0.95])
# ax.yaxis.labelpad = 25

plt.subplots_adjust(wspace=0.05, hspace=0.1)
fig = plt.gcf()
fig.text(0.03, 0.5, "AUC-ROC", rotation=90, ha="center", va="center", fontsize=18)
fig.text(0.5, 0.04, "Dimension, $K$", ha="center", va="center", fontsize=18)
fig.savefig(output_zoomed_file, dpi=300, bbox_inches="tight")

# %%
#
# Zoom out output
#
sns.set_style("white")
sns.set(font_scale=1.3)
sns.set_style("ticks")
sns.set_color_codes("deep")

col_order = list(data_labels.values())

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
    palette=palette,
    hue_order=hue_order,
    hue_kws=hue_kws,
    gridspec_kws={"wspace": 0},
)


g.map(sns.lineplot, "dim", "score", **plot_kwargs)
# g.map(sns.lineplot, "dim", "score", ci="sd")
for ax in g.axes:
    ax.axhline(0.5, ls="--", color="grey", alpha=0.8)

g.set(
    xticks=np.arange(3, 8),
    xticklabels=[r"$2^{}$".format(i) for i in range(3, 8)],
    yticks=np.linspace(0.2, 1, 5),
    yticklabels=[
        "%.1f" % d if d not in [0, 1] else ["0", "1"][int(d)]
        for d in np.linspace(0.2, 1, 5)
    ],
    ylim=(0.2, 1.01),
)

ax = g.axes[4]
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    columnspacing=1,
    fontsize=15,
)

g.set_xlabels("")
g.set_ylabels("")

subcap = "ABCDEFGHIJK"
g.set_titles(col_template="")
for i, ax in enumerate(g.axes):
    ax.annotate(
        (r"{\bf %s}" % subcap[i]) + r" {name}".format(name=col_order[i]),
        (0.05, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=14,
    )

plt.subplots_adjust(wspace=0.01, hspace=0.2)
fig = plt.gcf()
fig.text(0.03, 0.5, "AUC-ROC", rotation=90, ha="center", va="center", fontsize=18)
fig.text(0.5, 0.04, "Dimension, $K$", ha="center", va="center", fontsize=18)
fig.savefig(output_file, dpi=300, bbox_inches="tight")

# %%
