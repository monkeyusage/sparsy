import pandas as pd
import numpy as np

df = pd.read_csv("data/output.tsv", sep="\t")

df75mal = pd.read_csv("data/output_short70_mal_new_1975.tsv", sep="\t")
df75new = pd.read_csv("data/output_short70_new_1975.tsv", sep="\t")

sort_that = (df75mal, df75new)

df75mal.sort_values(["firm", "firm_"], inplace=True)
df75new.sort_values(["firm", "firm_"], inplace=True)

df["mal2"] = df75mal['maltec'].astype(np.uint64)
df["cov_mal2"] = df75mal['malcovtec'].astype(np.uint64)

df["std2"] = df75new["tec"]
df["cov_std2"] = df75new["covtec"]

cols = list(df.columns.values)
cols.sort()

df = df[cols]