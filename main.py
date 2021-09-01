"""
Parsy is a program that computes various distances between companies' carateristics
"""
import pandas as pd
import numpy as np
from numba import njit
from time import perf_counter

from sys import argv

def gen_data() -> np.ndarray:
    return np.array([
        np.random.randint(low=1, high=7000, size=15800000),
        np.random.randint(low=1, high=6, size=15800000)
    ]).T

@njit
def zero_diag(array: np.ndarray) -> np.ndarray:
    assert array.shape[0] == array.shape[1], "array must be square matrix"
    for i in range(array.shape[0]):
        array[i,i] = 0
    return array

@njit
def tclass_corr(tech: int, var:np.ndarray) -> np.ndarray:
    base_var = var.copy()
    for i in range(tech):
        for j in range(tech):
            var[i,j] = var[i,j] / (np.sqrt(base_var[i,i]) * np.sqrt(base_var[j,j]))
    return var

@njit
def firm_corr(num:int, tech:int, values:np.ndarray, base_std:np.ndarray) -> np.ndarray:
    norm_values = values.copy()    
    for j in range(tech):
        for i in range(num):
            norm_values[i,j] = values[i,j] / np.sqrt(base_std[i,i])
    return norm_values

def main() -> None:
    data = pd.read_csv(
        "data/data.tsv",
        sep="\t",
        usecols=["firm", "nclass"],
        dtype={
            "firm":np.uint16, # if company max int is above 65000 set it to uint32
            "nclass":"category"
        }
    )

    # sort by nclass and create a new tclass independant of naming of nclass just in case
    data = data.sort_values("nclass")
    tclass_replacements = dict((k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique())))
    data["tclass"] = data.nclass.replace(tclass_replacements)

    data = data.sort_values(["firm", "tclass"])

    # crosstab on firm and class
    subsh = pd.crosstab(data["firm"], data["tclass"]).astype(np.uint16)
    firms = subsh.index.values
    total = subsh.values.sum(axis=1)
    values = (subsh.values / total[:, None]) * 100
    tech = data["tclass"].nunique()

    del total, subsh, data

    num = values.shape[0]

    # matrix of correlations between classes
    var = np.dot(values.T, values)
    var = tclass_corr(tech, var)

    # correlation between firms overs classes (n x n || 694x694)
    base_std = np.dot(values, values.T)
    norm_values = firm_corr(num, tech, values, base_std)
    
    # generate standard measures
    std = (np.dot(norm_values, norm_values.T).round(decimals=2) * 100).astype(np.uint32)
    cov_std = (np.dot(values, values.T).round(decimals=2) * 100).astype(np.uint32)

    # generate MAL measure
    mal = (np.dot(np.dot(norm_values, var),norm_values.T).round(decimals=2) * 100).astype(np.uint64)
    cov_mal = (np.dot(np.dot(values, var), values.T).round(decimals=2) * 100).astype(np.uint64)

    # for the 4 metrics
    # sum up rows but before remove diagonal numbers (zero them out) so we don t count them twice
    # then log each one
    results = dict((name,zero_diag(arr).sum(axis=1)) for name,arr in {"std":std, "mal":mal, "cov_std":cov_std, "cov_mal":cov_mal}.items())

    del values, norm_values, var
    assert len(set([arr.shape[0] for arr in results.values()])) == 1, "arrays are not all the same length"

    # df creation for further saving
    df = pd.DataFrame({"firm":firms,**results})

    # saving into memory
    df.to_csv("data/spill_output_nolog.tsv", sep="\t", index=False)
    df.iloc[:,1:] = np.log(df.drop("firm", axis=1).values, dtype=np.float32)
    df.to_csv("data/spill_output_log.tsv", sep="\t", index=False)

if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")