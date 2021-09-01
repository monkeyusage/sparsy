"""
Parsy is a program that computes various distances between companies' carateristics
"""
import pandas as pd
import numpy as np
from time import perf_counter

from sys import argv

def main():
    try:
        file = argv[1]
    except IndexError:
        print("enter valid path for .csv analysis")
        exit()
    
    assert file.endswith(".tsv"), "data file format should be tsv"
    
    # read in the data file
    # uint8 => 255, uint16 => 65000 uint32 => 4294967295
    data = pd.read_csv(
        file,
        sep="\t",
        usecols=["firm", "nclass"],
        dtype={
            "firm":np.uint16, # if company max int is above 65000 set it to uint32
            "nclass":"category"
        }
    )
    
    del file

    # sort by nclass and create a new tclass independant of naming of nclass just in case
    data = data.sort_values("nclass")
    tclass_replacements = dict((k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique())))
    data["tclass"] = data.nclass.replace(tclass_replacements)
    
    # crosstab on firm and class
    subsh = pd.crosstab(data["firm"], data["tclass"]).astype(np.uint16)
    index = subsh.index.values.copy()
    total = subsh.values.sum(axis=1)
    values = (subsh.values / total[:, None]) * 100
    tech = data["tclass"].nunique()

    del total, subsh, data

    num = values.shape[0]

    # compute matrix of correlations between classes
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
    for i in range(num):
        for j in range(tech):
            norm_values[i,j] = values[i,j] / np.sqrt(base_std[i,i])
    
    # generate standard measures
    std = (np.dot(norm_values, norm_values.T).round(2) * 100).astype(np.uint8)
    cov_std = (np.dot(values, values.T).round(2) * 100).astype(np.uint32)

    # generate MAHALANOBIS measure ==> gives n x n matrix
    mal = (np.dot(np.dot(norm_values, var),norm_values.T).round(2) * 100).astype(np.uint8)
    cov_mal = (np.dot(np.dot(values, var), values.T).round(2) * 100).astype(np.uint32)

    # flatten arrays and store them in dict for pandas purpose
    results = dict((name,arr.flatten()) for name,arr in {"std":std, "mal":mal, "cov_std":cov_std, "cov_mal":cov_mal}.items())

    del values, norm_values, var, base_var

    assert len(set([arr.shape[0] for arr in results.values()])) == 1, "arrays are not all the same length"
    
    # firm indices
    firms = np.repeat(index, std.shape[0])
    firms_ = list(index) * std.shape[0]

    # df creation for further saving
    df = pd.DataFrame(
        {
            "firm":firms,
            "firm_":firms_,
            **results
        }
    )

    # saving into memory
    df.to_csv("data/output.tsv", sep="\t", index=False)

if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")