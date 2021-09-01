"""
Parsy is a program that computes various distances between companies' carateristics
"""
from __future__ import annotations
from typing import cast
import pandas as pd
import numpy as np
from numba import njit
from time import perf_counter
from json import load


def gen_data():
    return np.array([
        np.random.randint(low=1, high=500_000, size=15_800_000),
        np.random.randint(low=1, high=1000, size=15_800_000),
        np.random.randint(low=1970, high=2020, size=15_800_000)
    ]).T

@njit
def zero_diag(array):
    assert array.shape[0] == array.shape[1], "array must be square matrix"
    for i in range(array.shape[0]):
        array[i,i] = 0
    return array

@njit
def tclass_corr(tech: int, var):
    base_var = var.copy()
    for i in range(tech):
        for j in range(tech):
            var[i,j] = var[i,j] / (np.sqrt(base_var[i,i]) * np.sqrt(base_var[j,j]))
    return var

@njit
def firm_corr(num:int, tech:int, values, base_std):
    norm_values = values.copy()    
    for j in range(tech):
        for i in range(num):
            norm_values[i,j] = values[i,j] / np.sqrt(base_std[i,i])
    return norm_values

@njit
def dot_zero(array_a, array_b):
    dot_product = np.dot(array_a, array_b)
    rounded = np.empty_like(dot_product)
    np.round(dot_product, 2, rounded)
    multiplied = np.multiply(rounded, 100)
    zero_out = zero_diag(multiplied)
    summed = np.sum(zero_out, axis=1)
    logged = np.log(summed)
    return logged

@njit
def compute(matrix, tech: int):
    total = matrix.sum(axis=1)

    values = matrix
    # values= (matrix / total[:, None]) * 100
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            values[i,j] = (matrix[i,j] / total[i]) * 100

    num = values.shape[0]

    # compute matrix of correlations between classes
    var = np.dot(values.T, values)
    var = tclass_corr(tech, var)

    # correlation between firms overs classes (n x n || 694x694)
    base_std = np.dot(values, values.T)
    norm_values = firm_corr(num, tech, values, base_std)
    
    # generate standard measures
    std = dot_zero(norm_values, norm_values.T)
    cov_std = dot_zero(values, values.T)

    # generate MAHALANOBIS measure ==> gives n x n matrix
    mal = dot_zero(np.dot(norm_values, var),norm_values.T)
    cov_mal = dot_zero(np.dot(values, var), values.T)

    return std, cov_std, mal, cov_mal

def main() -> None:
    with open("data/config.json", "r") as config_file:
        config : dict[str, str | int] = load(config_file)

    input_file : str = cast(str, config["input_data"])
    data : pd.DataFrame =  pd.read_csv(
        input_file,
        sep="\t",
        usecols=["firm", "nclass", "year"],
        dtype={
            "firm":np.uint32,
            "nclass":"category",
            "year":"category"
        }
    ) if not config["stress_test"] else \
        pd.DataFrame(data=gen_data(), columns=("firm", "nclass", "year"))

    # sort by nclass and create a new tclass independant of naming of nclass just in case
    tclass_replacements = dict((k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique())))
    data["tclass"] = data.nclass.replace(tclass_replacements)

    # crosstab on firm and class
    subsh = pd.crosstab(data["firm"], data["tclass"]).astype(np.float32)
    firms = subsh.index.values.copy()

    std, cov_std, mal, cov_mal = compute(matrix=subsh.values, tech=data["nclass"].nunique())

    # df creation for further saving
    df = pd.DataFrame(
        {
            "year": year,
            "firm":firms,
            "std":std,
            "cov_std":cov_std,
            "mal":mal,
            "cov_mal":cov_mal
        }
    )
    
    # saving into memory
    df.to_csv("data/spill_output_log.tsv", sep="\t", index=False)

if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")