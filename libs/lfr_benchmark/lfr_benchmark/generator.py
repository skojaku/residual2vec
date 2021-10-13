"""This script generates the benchmark network using the LFR method. This
script calls the external program, which is installed as follows.

1. Download the source code. https://sites.google.com/site/andrealancichinetti/files/binary_networks.tar.gz?attredirects=0&d=1
2. Unpack and rename to lfr-generator.
3. go to the directory and compile by make

The following is the bash script to prepare the program

wget https://sites.google.com/site/andrealancichinetti/files/binary_networks.tar.gz
tar -xzf binary_networks.tar.gz
mv binary_networks lfr-generator
cd lfr-generator && make
rm binary_networks.tar.gz
"""

import json
import os
import pathlib
import random
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse


class NetworkGenerator:
    """A Python wrapper for LFR benchmark network generator.

        >>> lfr_params_list = {
        >>>     "N": 1000,
        >>>     "k": 20,
        >>>     "maxk": 50,
        >>>     "minc": 20,
        >>>     "maxc": 100,
        >>>     "tau2": 1,
        >>>     "tau": None,
        >>>     "mu": None,
        >>> }
        >>> tau1_list = [3, 6]
        >>> mu_list = np.linspace(0.05, 1, 20)
        >>>
        >>> generator =  NetworkGenerator()
        >>> networks = generator.generate(lfr_params_list, mu_list)
        >>> generator.save_networks(networks, "network-dir")
    """

    def __init__(self):
        pass

    def init_seed(self, path, seed):
        """Set the seed file for the external program."""
        with open("{path}/time_seed.dat".format(path=path), "w", encoding="utf-8") as f:
            f.write("%d" % seed)
            f.close()

    def generate_lfr_net(self, N, k, maxk, minc, maxc, tau, tau2, mu):
        """Generate the networks using the ./lfr-generator/benchmark program
        See ./lfr-generator/Readme.txt for the descriptions of the
        parameters."""

        root = pathlib.Path(__file__).parent.absolute()
        with tempfile.TemporaryDirectory() as tmpdirname:
            txt = "{root}/lfr-generator/benchmark -N {N} -k {k} -maxk {maxk} -t1 {t1} -t2 {t2} -mu {mu} -minc {minc} -maxc {maxc}".format(
                root=root,
                N=N,
                k=k,
                maxk=maxk,
                minc=minc,
                maxc=maxc,
                mu=mu,
                t1=tau,
                t2=tau2,
            )
            os.chdir(tmpdirname)
            seed = random.randint(0, 1e5)
            self.init_seed(tmpdirname, seed)

            os.system(txt)
            edges = pd.read_csv(
                "{tmp}/network.dat".format(tmp=tmpdirname), sep="\t", header=None
            ).values
            edges = edges - 1  # because the node id start from 1
            edges = pd.DataFrame(edges, columns=["source", "target"])

            communities = pd.read_csv(
                "{tmp}/community.dat".format(tmp=tmpdirname), sep="\t", header=None
            ).values
            communities[:, 0] -= 1  # because the node id start from 1
            communities = pd.DataFrame(communities, columns=["node_id", "community_id"])

        return edges, communities, seed

    def get_seed(self):
        """Get the seed value to generate the network."""
        f = open("time_seed.dat", "r", encoding="utf-8")
        seed = f.readline()
        f.close()
        return seed

    def generate(self, lfr_params, mu):
        params = lfr_params.copy()
        params["mu"] = mu

        edge_table, community_table, seed = self.generate_lfr_net(**params)

        N = community_table.shape[0]
        A = sparse.csr_matrix(
            (
                np.ones(edge_table.shape[0]),
                (edge_table["source"], edge_table["target"]),
            ),
            shape=(N, N),
        )

        networks = {
            "net": A,
            "community_table": community_table,
            "params": params,
            "seed": seed,
        }
        return networks

    def save_network(self, network, datadir):

        # Load the network
        A = network["net"]
        community_table = network["community_table"]
        params = network["params"]
        seed = network["seed"]

        # Create a directory
        os.makedirs(datadir, exist_ok=True)

        # Save
        sparse.save_npz("{datadir}/network.npz".format(datadir=datadir), A)
        community_table.to_csv(
            "{datadir}/community_table.tsv".format(datadir=datadir),
            sep="\t",
            index=False,
        )
        params["seed"] = seed
        with open("{datadir}/parameters.json".format(datadir=datadir), "w") as outfile:
            json.dump(params, outfile)
