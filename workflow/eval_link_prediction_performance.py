# %% import library
import pathlib
import sys
from glob import glob

import numpy as np
import pandas as pd

# import residual_node2vec as rv
import utils_link_pred
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# Helper Functions
def get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "=" not in p:
            continue
        kv = p.split("=")
        retval[kv[0]] = kv[1]
    return retval


# Loading
if "snakemake" in sys.modules:
    net_files = snakemake.input["net_files"]
    emb_files = snakemake.input["emb_files"]
    edge_files = snakemake.input["edge_files"]
    output_file = snakemake.output["output_file"]
else:
    net_files = [f for f in glob("../data/link-prediction/networks/net=*")]
    emb_files = [f for f in glob("../data/link-prediction/embeddings/*")]
    edge_files = [
        f for f in glob("../data/link-prediction/networks/test_edgelist_*.csv")
    ]
    output_file = "../data/link-prediction/results/auc_score.csv"


# %%
# Loading
#
emb_file_table = pd.DataFrame([get_params(r) for r in emb_files])
net_file_table = pd.DataFrame([get_params(r) for r in net_files])
edge_file_table = pd.DataFrame([get_params(r) for r in edge_files])

# %%
# Merging
#
emb_file_table = emb_file_table.rename(columns={"filename": "emb_file"})
edge_file_table = edge_file_table.rename(columns={"filename": "edge_file"})
net_file_table = net_file_table.rename(columns={"filename": "net_file"})
cols = list(set(emb_file_table.columns).intersection(set(edge_file_table.columns)))
file_table = pd.merge(emb_file_table, edge_file_table, on=cols)

cols = list(set(file_table.columns).intersection(set(net_file_table.columns)))
file_table = pd.merge(file_table, net_file_table, on=cols)

# %%
# Calculate the AUC
#


def calc_modeled_prob(emb, net, src, trg, model_name, membership, offset):
    dotsim = np.sum(emb[src, :] * emb[trg, :], axis=1)
    if model_name in [
        "deepwalk",
        "residual2vec-unbiased",
        "residual2vec-dotsim",
        "residual2vec-truncated-dotsim",
        "glove-dotsim",
        "jresidual2vec-unbiased",
        "node2vec-unbiased",
        "node2vec-qhalf",
        "node2vec-qdouble",
        "leigenmap",
        "netmf",
        "node2vec",
        "fairwalk",
        "levy-word2vec",
        "gcn",
        "gat",
        "graphsage",
        "gcn-doubleK",
        "graphsage-doubleK",
        "gat-doubleK",
    ]:
        return dotsim
    elif model_name == "glee":
        return -dotsim
    elif model_name == "glove":
        a, b = utils_link_pred.fit_glove_bias(net, emb)
        return dotsim + a[src] + b[trg]
    elif model_name in [
        "residual2vec",
        "jresidual2vec",
        "residual2vec-sim",
        "residual2vec-truncated",
        "lndeg",
    ]:
        # Modeled probability using degree
        outdeg = np.array(net.sum(axis=1)).reshape(-1)
        indeg = np.array(net.sum(axis=0)).reshape(-1)
        return (
            dotsim
            + np.log(np.maximum(indeg[src], 1))
            + np.log(np.maximum(outdeg[trg], 1))
        )
    elif model_name in ["residual2vec-adap"]:
        # Modeled probability using degree
        outdeg = np.array(net.sum(axis=1)).reshape(-1)
        indeg = np.array(net.sum(axis=0)).reshape(-1)
        return dotsim + offset[src] + offset[trg]


dg = file_table[
    file_table["model"].isin(["residual2vec", "glove", "residual2vec-truncated"])
]
dg["model"] += "-dotsim"
file_table = pd.concat([file_table, dg], ignore_index=True)

#
# Evaluation
#
def eval_link_pred(edge_file, df):
    results = []
    edge_table = pd.read_csv(edge_file).rename(columns={"0": "src", "1": "trg"})
    net = sparse.load_npz(df["net_file"].values[0])
    net = net + net.T
    for _i, row in df.iterrows():
        data = np.load(row["emb_file"])
        emb = data["emb"]
        membership = np.zeros(emb.shape[0])
        node_offset = np.zeros(emb.shape[0])
        if "membership" in data.keys():
            membership = data["membership"]
        if "offset" in data.keys():
            offset = data["offset"]
            node_offset = offset

        src, trg, y = edge_table["src"], edge_table["trg"], edge_table["edge_type"]
        n = emb.shape[0]
        s = (src < n) & (trg < n)
        src, trg, y = src[s], trg[s], y[s]
        likelihood = calc_modeled_prob(
            emb, net, src, trg, row["model"], membership, node_offset
        )

        node_offset = (
            np.log(np.maximum(1, np.array(net[:, :n][:n, :].sum(axis=0)))).reshape(-1)
            * node_offset
        )

        if any(np.isnan(likelihood)):
            score = 0.5
        else:
            score = roc_auc_score(y, likelihood)
        row["score"] = score
        results += [row]
    return results


# list_results = Parallel(n_jobs=1)(
list_results = Parallel(n_jobs=30)(
    delayed(eval_link_pred)(edge_file, df)
    for edge_file, df in tqdm(file_table.groupby("edge_file"))
)


#
# Merge
#
results = []
for res in list_results:
    results += res
result_table = pd.DataFrame(results)

#
# Save
#
# %%
result_table.to_csv(output_file, index=False)
