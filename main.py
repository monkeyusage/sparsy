"""
Parsy is a program that computes various distances between companies' carateristics
"""
import pandas as pd
import numpy as np
from time import perf_counter

from sys import argv

def zero_diag(array: np.ndarray) -> np.ndarray:
    for i in range(array.shape[0]):
        array[i,i] = 0
    return array

def main():
    try:
        file = argv[1]
    except IndexError:
        print("enter valid path for .csv analysis")
        exit()
    
    assert file.endswith(".tsv") or file.endswith(".dta"), "data file format should be either tsv or dta"
    
    # read in the data file
    data = pd.read_csv(file, sep="\t") if file.endswith(".tsv") else pd.read_stata(file)
    
    del file

    # set the appropriate data types
    data["division"] = data["division"].astype(np.uint64)
    data["year"] = data["year"].astype(np.uint16)
    data["firm"] = data["firm"].astype(np.uint16)
    data["nclass"] = data["nclass"].astype(np.uint8)

    # sort by nclass and create a new tclass independant of naming of nclass just in case
    data = data.sort_values("nclass")
    tclass_replacements = dict((k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique())))
    data["tclass"] = data.nclass.replace(tclass_replacements)

    data = data.sort_values(["firm", "tclass"])

    # crosstab on firm and class
    subsh = pd.crosstab(data["firm"], data["tclass"]).astype(np.uint16)
    index = subsh.index.values.copy()
    total = subsh.values.sum(axis=1)
    values = (subsh.values / total[:, None]) * 100
    tech = data["tclass"].nunique()

    # del total, subsh, data

    num = values.shape[0]

    # matrix of correlations between classes
    var = np.dot(values.T, values)
    base_var = var.copy()

    # var = tclass_corr(tech, var, base_var)
    for i in range(tech):
        for j in range(tech):
            var[i,j] = var[i,j] / (np.sqrt(base_var[i,i]) * np.sqrt(base_var[j,j]))

    # correlation between firms overs classes (n x n || 694x694)
    base_std = np.dot(values, values.T)
    
    # norm_values = firm_corr(num, tech, values, base_std)
    norm_values = values.copy()
    for j in range(tech):
        for i in range(num):
            norm_values[i,j] = values[i,j] / np.sqrt(base_std[i,i])
    
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

    # del values, norm_values, var, base_var
    assert len(set([arr.shape[0] for arr in results.values()])) == 1, "arrays are not all the same length"

    # df creation for further saving
    df = pd.DataFrame(results)

    # saving into memory
    df.to_csv("data/spill_output_nolog.tsv", sep="\t", index=False)
    pd.DataFrame(
        data=np.log(df.values).astype(np.float16), columns=df.columns
    ).to_csv("data/spill_output_log.tsv", sep="\t", index=False)
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")