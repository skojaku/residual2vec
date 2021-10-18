import itertools
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.metrics import average_precision_score

from .generator import NetworkGenerator as NetworkGenerator


class Benchmark:
    def __init__(self, dim=64):
        self.dim = dim
        self.mu_list = np.linspace(0.05, 1, 20)

    def run(self, lfr_params, populate_model):

        root = Path().parent.absolute()

        ng = NetworkGenerator()
        networks = ng.generate(lfr_params, self.mu_list)

        os.chdir(root)

        results = []
        for data in networks:
            net = data["net"]
            community_table = data["community_table"]
            params = data["params"]
            seed = data["seed"]

            task = TaskLFR(self.dim)
            task.run(net, community_table, populate_model)

            # Append parameters
            for i in range(len(task.result)):
                task.result[i]["mu"] = params["mu"]
                task.result[i]["seed"] = seed

            results += task.result

        return results


class TaskLFR:
    def __init__(self, dim=64):
        self.dim = dim
        self.result = []

    def run(self, net, community_table, populate_model):

        # Populate models
        model_list = populate_model()
        for model_name, model in model_list.items():

            # Training
            model.fit(net)
            invec = model.transform(self.dim)
            outvec = model.transform(self.dim, return_out_vector=True)

            # Evaluate the goodness for the in-vector
            sim = invec @ invec.T
            q = self.eval(sim, community_table)
            self.result += [
                {
                    "q": q,
                    "model": type(model).__name__,
                    "model_name": model_name,
                    "sim_type": "in-vector",
                }
            ]

            sim = outvec @ outvec.T
            q = self.eval(sim, community_table)
            self.result += [
                {
                    "q": q,
                    "model": type(model).__name__,
                    "model_name": model_name,
                    "sim_type": "out-vector",
                }
            ]

    def eval(self, sim, community_table):
        N = np.minimum(sim.shape[0], community_table.shape[0])
        community_table = community_table.head(N)

        U = sparse.csc_matrix(
            (np.ones(N), (np.arange(N), community_table.community_id)), shape=(N, N)
        )
        sim_target = (U @ U.T).toarray() > 0
        sim_target = sim_target.reshape(-1)
        sim = np.array(sim).reshape(-1)
        ave_prec = average_precision_score(sim_target, sim)
        return ave_prec

    def save(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.result, outfile)
