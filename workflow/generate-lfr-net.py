import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.append(os.path.abspath(os.path.join("./libs/lfr_benchmark")))
from lfr_benchmark.generator import NetworkGenerator as NetworkGenerator

PARAM_FILE = snakemake.input["param_file"]
param_id = snakemake.params["param_id"]
mu = snakemake.params["mu"]

OUTPUT_NET_FILE = snakemake.output["output_net_file"]
OUTPUT_COMMUNITY_FILE = snakemake.output["output_community_file"]
OUTPUT_PARAM_FILE = snakemake.output["output_param_file"]

with open(PARAM_FILE, "rb") as f:
    lfr_params = json.load(f)

params = lfr_params[param_id]

root = Path().parent.absolute()
ng = NetworkGenerator()
data = ng.generate(params, mu)
os.chdir(root)

# Load the network
net = data["net"]
community_table = data["community_table"]
params = data["params"]
seed = data["seed"]

# Save
sparse.save_npz(OUTPUT_NET_FILE, net)
community_table.to_csv(
    OUTPUT_COMMUNITY_FILE, index=False,
)
params["seed"] = seed
with open(OUTPUT_PARAM_FILE, "w") as outfile:
    json.dump(params, outfile)
