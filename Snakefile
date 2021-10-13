from os.path import join as j
from itertools import product
import numpy as np


configfile: "workflow/config.yaml"


DATA_DIR = config["data_dir"]
FIG_DIR = config["fig_dir"]


# ====================
# Link Prediction
# ====================

#
# Source
#
LP_DATA_SRC = {
    "ca-AstroPh": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    "ca-HepTh": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "dblp_cite": "https://networks.skewed.de/net/dblp_cite",
    "openflights": "https://networks.skewed.de/net/openflights",
    "polblogs": "https://networks.skewed.de/net/polblogs",
    "ppi": "https://snap.stanford.edu/node2vec/Homo_sapiens.mat",
    "wiki-vote": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
}

#
# Preprocessed
#
LP_DIR = j(DATA_DIR, "link-prediction")
LP_EDGE_LIST_FILE = j(LP_DIR, "preprocessed", "{network}.csv")
PPI_NET_MAT_FORMAT = j(LP_DIR, "preprocessed", "ppi.mat")
# LP_EDGE_LIST_FILE_ALL = expand(LP_EDGE_LIST_FILE, network=LP_NET_LIST)


#
# Splitting the edges into train and test sets
#
split_params = {
    "index": range(1),
    #"index": range(30),
    "frac": 0.5,  # fraction of edges removed
    "directed": "undirected",  # directed and undireced netwoek must be sperated.
}
LP_SPLITTED_NET_FILE = j(
    LP_DIR, "benchmark-data", "net={network}_d={directed}_frac={frac}_index={index}.npz"
)
LP_SPLITTED_EDGE_FILE = j(
    LP_DIR,
    "benchmark-data",
    "test_edgelist_net={network}_d={directed}_frac={frac}_index={index}.csv",
)

#
# Embedding
#
LP_EMB_FILE_DIR = j(LP_DIR, "embeddings")
emb_params = {
    "model_name": [
        "node2vec",
        "node2vec-qhalf",
        "node2vec-qdouble",
        "deepwalk",
        "glove",
        "residual2vec",
        "netmf",
        "leigenmap",
        "graphsage",
        "gat",
        "gcn",
        "graphsage-doubleK",
        "gcn-doubleK",
        "gat-doubleK",
        "lndeg",
        "fairwalk",
    ],
    "window_length": [10],
    "dim": [8, 16, 32, 64, 128],
}
LP_EMB_FILE = j(
    LP_EMB_FILE_DIR,
    "emb_net={network}_d={directed}_frac={frac}_index={index}_model={model_name}_wl={window_length}_dim={dim}.npz",
)

#
# Results
#
LP_RESULT = j(LP_DIR, "results", "auc_score.csv")
LP_RESULT_FIG = j(FIG_DIR, "result-link-prediction.pdf")
LP_RESULT_SI_FIG = j(FIG_DIR, "result-link-prediction-gcn-graphsage.pdf")

FIGS = [LP_RESULT_FIG, LP_RESULT_SI_FIG]


# ==================
# LFR-benchmark
# ==================

LFR_DIR = j(DATA_DIR, "lfr")

#
# Source
#
LFR_GENERATOR = "libs/lfr_benchmark/lfr_benchmark/lfr-generator"
LFR_CONFIG_FILE = j(LFR_DIR, "lfr-config.json")


#
# Networks
#
LFR_NET_FILE = j(
    LFR_DIR, "networks", "param_id={param_id}_mu={mu}_s={sample_id}_network.npz"
)
LFR_COMMUNITY_FILE = j(
    LFR_DIR, "networks", "param_id={param_id}_mu={mu}_s={sample_id}_community_table.csv"
)
LFR_PARAM_FILE = j(
    LFR_DIR, "networks", "param_id={param_id}_mu={mu}_s={sample_id}_params.json"
)

LFR_PARAMS = {
    "mu": ["%.2f" % x for x in np.linspace(0.05, 1, 5)],
    #"mu": ["%.2f" % x for x in np.linspace(0.05, 1, 20)],
    "param_id": ["%d" % x for x in [0, 1]],
    "sample_id": ["%d" % x for x in np.arange(5)],
    #"sample_id": ["%d" % x for x in np.arange(30)],
}

#
# Embeddings
#
LFR_EMB_PARAMS = {
    "model_name": [
        "node2vec",
        "node2vec-qdouble",
        "node2vec-qhalf",
        "deepwalk",
        "glove",
        "residual2vec",
        "netmf",
        "leigenmap",
        "graphsage",
        "gcn",
        "gat",
        "graphsage-doubleK",
        "gcn-doubleK",
        "gat-doubleK",
        "fairwalk",
    ],
    "window_length": [10],
    "dim": [64],
}

LFR_EMB_FILE = j(
    LFR_DIR,
    "embeddings",
    "emb_net_param_id={param_id}_mu={mu}_s={sample_id}_model={model_name}_wl={window_length}_dim={dim}.npz",
)

#
# Results
#
LFR_RESULT_FILE = j(LFR_DIR, "results", "results.csv")
LFR_RESULT_FIG = j(FIG_DIR, "result-lfr.pdf")
LFR_RESULT_ZOOMED_FIG = j(FIG_DIR, "result-zoomed-lfr.pdf")
LFR_RESULT_SI_FIG = j(FIG_DIR, "result-lfr-gcn-graphsage.pdf")

FIGS += [LFR_RESULT_FIG, LFR_RESULT_ZOOMED_FIG, LFR_RESULT_SI_FIG]

# ===================
# Block approximation
# ===================

BA_NET_LIST = [
    "ca-AstroPh",
    "ca-HepTh",
    "dblp_cite",
    "openflights",
    "polblogs",
    "wiki-vote",
    "ppi",
]
BA_RES_FILE = j(DATA_DIR, "test-block-approximation", "yypred_net={network}.csv",)
BA_RESULT_FIG = j(FIG_DIR, "result-sbm-approximation.pdf")
FIGS += [BA_RESULT_FIG]


#
# Rule to create all figures
#
rule all:
    input:
        FIGS,


# =================
# Link prediction
# =================


#
# Download networks
#
rule download_from_snap:
    params:
        url=lambda wildcards: LP_DATA_SRC[wildcards.network],
    output:
        output_file=LP_EDGE_LIST_FILE,
    wildcard_constraints:
        network="(" + ")|(".join(["ca-AstroPh", "ca-HepTh", "wiki-vote"]) + ")",
    script:
        "workflow/download-net-from-snap.py"


rule download_ppi_net_in_mat_format:
    params:
        url=LP_DATA_SRC["ppi"],
    output:
        output_file=PPI_NET_MAT_FORMAT,
    shell:
        r"""
                wget {params.url} -O {output.output_file}
                """


rule mat2edgelist:
    input:
        input_file=PPI_NET_MAT_FORMAT,
    output:
        output_file=LP_EDGE_LIST_FILE.format(network="ppi"),
    wildcard_constraints:
        network="(" + ")|(".join(["ppi"]) + ")",
    script:
        "workflow/mat2edgelist.py"


rule download_from_netzschleuder:
    params:
        net_name=lambda wildcards: wildcards.network,
    output:
        output_file=LP_EDGE_LIST_FILE,
    wildcard_constraints:
        network="(" + ")|(".join(["dblp_cite", "openflights", "polblogs"]) + ")",
    script:
        "workflow/download-net-from-netzschleuder.py"


#
# Embedding
#


rule train_test_set_split:
    input:
        edge_file=LP_EDGE_LIST_FILE,
    output:
        net_file=LP_SPLITTED_NET_FILE,
        test_edge_file=LP_SPLITTED_EDGE_FILE,
    params:
        removal_frac=lambda wildcards: wildcards.frac,
        directed=lambda wildcards: wildcards.directed,
    script:
        "workflow/train_test_set_split.py"


rule embedding_link_prediction:
    input:
        netfile=LP_SPLITTED_NET_FILE,
    output:
        embfile=LP_EMB_FILE,
    params:
        model_name=lambda wildcards: wildcards.model_name,
        dim=lambda wildcards: wildcards.dim,
        window_length=lambda wildcards: wildcards.window_length,
        directed=lambda wildcards: wildcards.directed,
        num_walks=10,
    script:
        "workflow/embedding.py"


#
# Evaluation
#


rule eval_link_prediction:
    input:
        net_files=expand(
            LP_SPLITTED_NET_FILE, network=LP_DATA_SRC.keys(), **split_params
        ),
        emb_files=expand(
            LP_EMB_FILE, **emb_params, network=LP_DATA_SRC.keys(), **split_params
        ),
        edge_files=expand(
            LP_SPLITTED_EDGE_FILE, network=LP_DATA_SRC.keys(), **split_params
        ),
    output:
        output_file=LP_RESULT,
    script:
        "workflow/eval_link_prediction_performance.py"


#
# Plot
#


rule plot_result_link_prediction:
    input:
        res_file=LP_RESULT,
    output:
        output_file=LP_RESULT_FIG,
    script:
        "workflow/plot_result_link_prediction.py"


rule plot_result_link_prediction_SI:
    input:
        res_file=LP_RESULT,
    output:
        output_file=LP_RESULT_SI_FIG,
    script:
        "workflow/plot_result_link_prediction-gcn-graphsage.py"


# =================
# LFR bechmark
# =================

#
# Prep. Generator
#


rule setup_lfr_benchmark_generator:
    output:
        LFR_GENERATOR,
    shell:
        "wget -c https://sites.google.com/site/andrealancichinetti/files/binary_networks.tar.gz -O - | tar -xz && mv binary_networks {output} && cd {output};make"


#
# Networks
#


rule generate_lfr_net:
    input:
        param_file=LFR_CONFIG_FILE,
        generator=LFR_GENERATOR,
    output:
        output_net_file=LFR_NET_FILE,
        output_community_file=LFR_COMMUNITY_FILE,
        output_param_file=LFR_PARAM_FILE,
    params:
        param_id=lambda wildcards: int(wildcards.param_id),
        mu=lambda wildcards: wildcards.mu,
    script:
        "workflow/generate-lfr-net.py"


#
# Embeddings
#


rule embedding_lfr_net:
    input:
        netfile=LFR_NET_FILE,
    output:
        embfile=LFR_EMB_FILE,
    params:
        model_name=lambda wildcards: wildcards.model_name,
        dim=lambda wildcards: wildcards.dim,
        window_length=lambda wildcards: wildcards.window_length,
        directed="undirected",
        num_walks=10,
    script:
        "workflow/embedding.py"


#
# Evaluation
#


rule eval_lfr_benchmark_performance:
    input:
        emb_files=expand(LFR_EMB_FILE, **LFR_PARAMS, **LFR_EMB_PARAMS),
        com_files=expand(LFR_COMMUNITY_FILE, **LFR_PARAMS, **LFR_EMB_PARAMS),
    output:
        output_file=LFR_RESULT_FILE,
    script:
        "workflow/eval_lfr_benchmark_performance.py"


#
# Plot
#


rule plot_result_lfr:
    input:
        result_file=LFR_RESULT_FILE,
        config_file=LFR_CONFIG_FILE,
    output:
        output_file=LFR_RESULT_FIG,
        output_zoomed_file=LFR_RESULT_ZOOMED_FIG,
    script:
        "workflow/plot-lfr-result.py"


rule plot_result_lfr_SI:
    input:
        result_file=LFR_RESULT_FILE,
        config_file=LFR_CONFIG_FILE,
    output:
        output_file=LFR_RESULT_SI_FIG,
    script:
        "workflow/plot-lfr-result-gcn-graphsage.py"


# ===================
# Block approximation
# ===================


rule test_sbm_trans_prob_approx:
    input:
        edge_file=LP_EDGE_LIST_FILE,
    params:
        approx_order=1,
    output:
        output_file=BA_RES_FILE,
    script:
        "workflow/test-block-approximation.py"


rule plot_sbm_trans_prob_approx:
    input:
        input_files=expand(BA_RES_FILE, network=BA_NET_LIST),
    output:
        output_file=BA_RESULT_FIG,
    script:
        "workflow/plot-test-sbm-approximation.py"
