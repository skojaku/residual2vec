import csv
import errno
import logging
import os

import networkx as nx
import numpy as np
from texttable import Texttable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def tab_printer(args):
    """Function to print the logs in a nice tabular format.

    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    print(t.draw())


def mk_outdir(out_path):
    """Check and make a directory.

    :param out_path: path for directory
    """
    if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    logging.info("output directory is created")


def read_graph(input_file_path, weighted=False, directed=False):
    """Reads the input network and return a networkx Graph object.

    :param input_file_path: File path of input graph
    :param weighted: weighted network(True) or unweighted network(False)
    :param directed: directed network(True) or undirected network(False)
    :return G: output network
    """
    if weighted:
        G = nx.read_edgelist(input_file_path, nodetype=str, create_using=nx.DiGraph(),)
    else:
        G = nx.read_edgelist(input_file_path, nodetype=str, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    if not directed:
        G = G.to_undirected()

    return G


def read_edge_file(file_path):
    """Read a edge list for link prediction (test or negative edges)

    :param file_path: Path of edge lists (.tsv format)
    :return: edge list
    """
    with open(file_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(map(tuple, reader))
    return data


def alias_setup(probs):
    """Utility function for random walker Compute utility lists for non-uniform
    sampling from discrete distributions.

    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """Draw sample from a non-uniform discrete distribution using alias
    sampling."""
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
