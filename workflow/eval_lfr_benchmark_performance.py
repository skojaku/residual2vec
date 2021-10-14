# %% import library
import pathlib

import faiss
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm


class FaissKNN:
    def __init__(self, k=10):
        self.k = k
        return

    def fit(self, X, y):
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(np.ascontiguousarray(X.astype(np.float32)))
        self.y = y.copy()
        self.index = index
        return self

    def set_params(self):
        pass

    def get_params(self):
        return None

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        ypred = np.array(stats.mode(self.y[indices], axis=1)[0]).reshape(-1)
        return ypred

    def score(self, X, y):
        ypred = self.predict(X)
        return f1_score(y, ypred, average="macro")


def calc_nmi(y, ypred):
    _, y = np.unique(y, return_inverse=True)
    _, ypred = np.unique(ypred, return_inverse=True)

    K = len(set(y))
    N = len(y)
    U = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N, K))
    Upred = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K)
    )
    prc = np.array((U.T @ Upred).toarray())
    prc = prc / np.sum(prc)
    pr = np.array(np.sum(prc, axis=1)).reshape(-1)
    pc = np.array(np.sum(prc, axis=0)).reshape(-1)

    # Calculate the mutual information
    Irc = stats.entropy(prc.reshape(-1), np.outer(pr, pc).reshape(-1))

    # Normalize MI
    Q = 2 * Irc / (stats.entropy(pr) + stats.entropy(pc))
    return Q


def calc_esim(y, ypred):
    _, y = np.unique(y, return_inverse=True)
    _, ypred = np.unique(ypred, return_inverse=True)

    K = len(set(y))
    M = len(y)
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(M, K))
    UB = sparse.csr_matrix(
        (np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(M, K)
    )

    fA = np.array(UA.sum(axis=0)).reshape(-1)
    fB = np.array(UB.sum(axis=0)).reshape(-1)
    fAB = (UA.T @ UB).toarray()

    Si = (
        0.5
        * fAB[(y, ypred)]
        * (1.0 / fA[y] + 1.0 / fB[ypred] - np.abs(1.0 / fA[y] - 1.0 / fB[ypred]))
    )
    S = np.mean(Si)
    return S


# %%
emb_files = snakemake.input[
    "emb_files"
]  # list(glob.glob("../data/lfr-benchmark/embeddings/*"))
com_files = snakemake.input[
    "com_files"
]  # list(glob.glob("../data/lfr-benchmark/networks/*community*"))
output_file = snakemake.output["output_file"]

#
# Load
#
def get_params(filename):
    params = pathlib.Path(filename).stem.split("_")
    retval = {"filename": filename}
    for p in params:
        if "=" not in p:
            continue
        kv = p.split("=")
        retval[kv[0]] = kv[1]
    return retval


emb_file_table = pd.DataFrame([get_params(r) for r in emb_files])
com_file_table = pd.DataFrame([get_params(r) for r in com_files])

emb_file_table = emb_file_table.rename(
    columns={"filename": "emb_file", "id": "param_id"}
)
com_file_table = com_file_table.rename(
    columns={"filename": "com_file", "id": "param_id"}
)

cols = list(set(emb_file_table.columns).intersection(set(com_file_table.columns)))
file_table = pd.merge(emb_file_table, com_file_table, on=cols)

#
# Evaluation
#


def auc_pred_groups(emb, group_ids, sample=100000, iterations=5):

    clabels, group_ids = np.unique(group_ids, return_inverse=True)
    num_nodes = emb.shape[0]

    #
    # Sample node pairs
    #
    def sample_node_pairs():
        nodepairs = []
        # Randomly sample node pairs within the community
        n0, n1 = [], []
        n0 = np.random.randint(0, num_nodes, sample)
        g0 = group_ids[n0]
        gcount = np.bincount(g0)
        for gid, cnt in enumerate(gcount):
            n1 += [np.random.choice(np.where(group_ids == gid)[0], cnt, replace=True)]
        n1 = np.concatenate(n1)
        g1 = group_ids[n1]

        o0, o1 = np.argsort(g0), np.argsort(g1)
        n0, n1 = n0[o0], n1[o1]
        g0, g1 = g0[o0], g1[o1]  # for testing
        nodepairs = [pd.DataFrame({"n0": n0, "n1": n1, "class": 1})]

        # Randomly sample negative pairs between the community
        sampled = 0
        while sampled < sample:
            n0 = np.random.randint(0, num_nodes, sample - sampled)
            n1 = np.random.randint(0, num_nodes, sample - sampled)
            dif_groups = np.where(group_ids[n0] != group_ids[n1])[0]
            n0, n1 = n0[dif_groups], n1[dif_groups]
            sampled += len(dif_groups)
            nodepairs += [pd.DataFrame({"n0": n0, "n1": n1, "class": 0})]
        nodepairs = pd.concat(nodepairs)
        return nodepairs["n0"].values, nodepairs["n1"].values, nodepairs["class"].values

    def sample_node_pairs_2():
        n0 = np.random.randint(0, num_nodes, sample)
        n1 = np.random.randint(0, num_nodes, sample)
        g0, g1 = group_ids[n0], group_ids[n1]
        y = np.array(g0 == g1).astype(int)
        return n0, n1, y

    #
    # Evaluation
    #
    def eval_auc(emb, n1, n0, y):
        e0 = emb[n0, :]
        e1 = emb[n1, :]
        dotsim = np.sum(np.multiply(e0, e1), axis=1)
        return metrics.roc_auc_score(y, dotsim)

    #
    # Main
    #
    # from sklearn.linear_model import LogisticRegression
    #    X = emb.copy()
    #    y = group_ids.copy().astype(int)
    #    score = []
    #    kf = KFold(n_splits=10)
    #    for train_index, test_index in kf.split(X, y):
    #        X_train, X_test = X[train_index, :], X[test_index, :]
    #        y_train, y_test = y[train_index], y[test_index]
    #        # log_reg = LogisticRegression()
    #        # score = cross_val_score(log_reg, emb, group_ids.astype(int), cv=10)
    #        score += [FaissKNN().fit(X_train, y_train).score(X_test, y_test)]
    #        # score = cross_val_score(model, emb, group_ids.astype(int), cv=10)
    score = []
    for _ in range(iterations):
        n1, n0, y = sample_node_pairs_2()
        score += [eval_auc(emb, n1, n0, y)]
    return np.mean(score)


def eval_clu(com_file, df):
    # Load community table
    com_table = pd.read_csv(com_file)
    y = com_table.community_id
    # K = len(set(com_table.community_id))

    # Load emebdding
    emb_list = {}
    for _i, row in df.iterrows():
        emb = np.load(row["emb_file"])["emb"]
        emb = emb.copy(order="C").astype(np.float32)
        emb_list[row["emb_file"]] = emb

    # Evaluate
    results = []
    for metric in ["cosine", "euclidean"]:
        for _i, row in df.copy().iterrows():
            emb = emb_list[row["emb_file"]]
            X = emb.copy()
            if metric == "cosine":
                X = np.einsum(
                    "ij,i->ij", X, 1 / np.maximum(np.linalg.norm(X, axis=1), 1e-12)
                )
            score = auc_pred_groups(X, y, iterations=1)
            row["auc"] = score
            row["metric"] = metric
            results += [row]
    return results


list_results = [eval_clu(com_file, df) for com_file, df in tqdm(file_table.groupby("com_file"))]

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
result_table.to_csv(output_file, index=False)

# %%
