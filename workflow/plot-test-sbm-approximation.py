# %%
import glob
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics


def get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "=" not in p:
            continue
        kv = p.split("=")
        kv[0] = kv[0].replace("yypred-", "")
        retval[kv[0]] = kv[1]
    return retval


if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
else:
    input_files = list(glob.glob("../data/sbm-trans-prob-approx/approx=2/*.csv"))
    output_file = "../figs/result-sbm-approximation.pdf"

file_table = pd.DataFrame([get_params(f) for f in input_files])
dflist = []
for _, row in file_table.iterrows():
    net_name = row["net"]
    df = pd.read_csv(row["filename"])
    df["net"] = row["net"]
    dflist += [df]
df = pd.concat(dflist, ignore_index=True)
# %%
rss = []
for _, dg in df.groupby(["net", "window_length"]):
    rs, _ = stats.pearsonr(dg["ypred"], dg["y"])
    rss += [rs]
np.mean(rss)

# %%

# %%
def hexbin(x, y, **kwargs):
    rs, _ = stats.pearsonr(x, y)
    # rs = metrics.r2_score(x, y)

    xmin = np.quantile(x, 0.0)
    xmax = np.quantile(x, 1)

    plt.hexbin(
        x,
        y,
        gridsize=50,
        edgecolors="none",
        cmap="Greys",
        linewidths=0.1,
        mincnt=5,
        **kwargs
    )
    ax = plt.gca()
    ax.plot(
        [xmin, xmax], [xmin, xmax], color=sns.color_palette().as_hex()[3], lw=3, ls=":"
    )

    xmin = np.quantile(x, 0.01)
    xmax = np.quantile(x, 1 - 0.001)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.annotate(
        r"$\rho=%.2f$" % rs,
        (0.05, 1 - 0.05),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=20,
    )

    return ax


# %%
row_order = df["net"].drop_duplicates().values

row_labels = {
    "openflights": "Airport",
    "ppi": "Protein-Protein",
    "wiki-vote": "Wikipedia vote",
    "ca-HepTh": "Coauthorship (HepTh)",
    "ca-AstroPh": "Coauthorship (AstroPh)",
    # "polblogs": "Blog",
    "dblp": "Citation (DBLP)",
}

df["net_name"] = df["net"].apply(lambda x: row_labels.get(x, np.nan))
col_order = df["window_length"].drop_duplicates().values

g = sns.FacetGrid(
    data=df,
    row="net_name",
    col="window_length",
    sharey=False,
    sharex=False,
    aspect=1.3,
    col_order=col_order,
    row_order=row_labels.values(),
)
g.map(hexbin, "y", "ypred")
g.set_ylabels("")
g.set_xlabels("")
g.set_titles("")
# g.set_ylabels(r"Approximated $T_{ij}$")
# g.set_xlabels(r"$T_{ij}$")
g.fig.text(
    -0.015,
    0.5,
    r"Approximated $\log P_{{\rm d}}(j \vert i)$",
    fontsize=20,
    rotation=90,
    ha="center",
    va="center",
)
g.fig.text(
    0.5,
    -0.01,
    r"Exact $\log P_{{\rm d}}(j \vert i)$",
    fontsize=20,
    ha="center",
    va="center",
)

for i, rlabel in enumerate(row_labels.values()):
    g.axes[i, 0].set_ylabel(rlabel, fontsize=15)
    g.axes[i, 0].get_yaxis().set_label_coords(-0.2, 0.5)

for i, clabel in enumerate(col_order):
    g.axes[0, i].set_title("Window size T=%d" % clabel, fontsize=15)

g.fig.subplots_adjust(wspace=0.2, hspace=0.1)
plt.tight_layout()
g.fig.savefig(output_file, dpi=300, bbox_inches="tight")
# %%
