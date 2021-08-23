import pandas as pd
import numpy as np

from sys import argv

file = argv[1]
if file in ("", None):
    print("enter valid path")
    exit()

data = pd.read_csv(file, sep="\t")
data["division"] = data["division"].astype(np.uint64)
data["year"] = data["year"].astype(np.uint16)
data["nclass"] = data["nclass"].astype(np.uint8)
data["firm"] = data["firm"].astype(np.uint16)

data = data.sort_values("nclass")
tclass_replacements = dict((k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique())))
data["tclass"] = data.nclass.replace(tclass_replacements)
tech = data["nclass"].nunique()

data = data.sort_values(["firm", "tclass"])
tclass_max = data["tclass"].max()

subsh = pd.crosstab(data["firm"], data["tclass"]).astype(np.uint32)
total = subsh.values.sum(axis=1)
idx = subsh.index.values
subsh = (subsh.values / total[:, None]) * 100
subsh = pd.DataFrame({"firm":idx, **{f"subsh{idx}":subsh[:,idx] for idx in range(subsh.shape[1])}})
