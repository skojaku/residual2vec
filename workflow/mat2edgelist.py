import pandas as pd
import scipy.io
from scipy import sparse

input_file = snakemake.input["input_file"]
output_file = snakemake.output["output_file"]

src, trg, _ = sparse.find(scipy.io.loadmat(input_file)["network"])
pd.DataFrame({"src": src, "trg": trg}).to_csv(output_file, index=False)
