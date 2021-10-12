from __future__ import annotations

import logging
from json import load
from pathlib import Path
from sys import argv
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from sparsy.numeric import dot_zero, mahalanobis
from sparsy.utils import chunker, reduce_data

# Those are function definitions of now memory optimised cython functions
# We keep them here for comparision's sake
# def dot_zero_old(matrix:np.ndarray) -> np.ndarray:
#     out = matrix.dot(matrix.T) * 100
#     np.fill_diagonal(out, 0)
#     out = out.sum(axis=1)
#     return out

# def mahalanobis_old(biggie:np.ndarray, small:np.ndarray) -> np.ndarray:
#     out = biggie.dot(small.dot(biggie.T))
#     out = np.round(out, decimals=2) * 100
#     np.fill_diagonal(out, 0)
#     out = out.sum(axis=1)
#     return out


def tclass_corr(values: np.ndarray) -> np.ndarray:
    var: np.ndarray = values.T.dot(values)
    base_var = var.copy()
    for i in range(var.shape[0]):
        for j in range(var.shape[0]):
            if var[i, i] == 0 or var[j, j] == 0:
                continue
            var[i, j] = var[i, j] / (np.sqrt(base_var[i, i]) * np.sqrt(base_var[j, j]))
    return var


def compute(matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    values: np.ndarray = ((matrix / matrix.sum(axis=1)[:, None]) * 100).astype(
        "float32"
    )
    # compute matrix of correlations between classes (m x m)
    var = tclass_corr(values)

    # correlation between firms overs classes (n x n)
    logging.info("most cpu intensive tasks now")

    # np.dot(arr, arr.T).diagonal() == (arr * arr).sum(axis=1)
    norm_values = values / np.sqrt((values * values).sum(axis=1))[:, None]

    # generate standard measures
    std = dot_zero(norm_values)
    cov_std = dot_zero(values)
    # generate MAHALANOBIS measure
    mal = mahalanobis(norm_values, var.dot(norm_values.T))
    cov_mal = mahalanobis(values, var.dot(values.T))

    return std, cov_std, mal, cov_mal


def preprocess(
    dataframe: pd.DataFrame, sub_years: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    df = dataframe[dataframe["year"].isin(set(sub_years))]
    if df.empty:
        return None
    # crosstab on firm and class
    firms: np.ndarray
    subsh: np.ndarray

    ct = (
        df.groupby(["firm", "tclass"], as_index=False)
        .size()
        .pivot("firm", "tclass", "size")
        .fillna(0)
        .astype(np.uint16)
    )
    firms = ct.index
    subsh = ct.to_numpy()
    return firms, subsh


def post_process(
    firms: np.ndarray,
    year: int,
    std: np.ndarray,
    cov_std: np.ndarray,
    mal: np.ndarray,
    cov_mal: np.ndarray,
    outfile: Path,
) -> None:
    df = pd.DataFrame(
        {
            "firm": firms,
            "year": year,
            "spilltec": std,
            "spillcovtec": cov_std,
            "spillmaltec": mal,
            "spillmalcovtec": cov_mal,
        }
    )
    # saving into memory into tmp.tsv files
    tmpfile = outfile.parent / f"{year}_tmp.tsv"
    df.to_csv(tmpfile, sep="\t", index=False)


def process(data: pd.DataFrame, year_set: list[int], outfile: Path) -> None:
    sub_data = preprocess(data, year_set)
    if sub_data is None:
        return
    year = max(year_set)
    firms, subsh = sub_data

    std, cov_std, mal, cov_mal = compute(subsh)

    if outfile != Path(""):
        # df creation for further saving
        post_process(firms, year, std, cov_std, mal, cov_mal, outfile)


def core(data: pd.DataFrame, iter_size: int, outfile: Path) -> None:
    years: list[int] = list(range(data["year"].min(), data["year"].max() + 1))
    logging.info("launching main process on one process")

    for year_set in tqdm(chunker(years, iter_size)):
        logging.info("processing data using year: %s", year_set)
        maybe = preprocess(data, year_set)
        if maybe is None:
            continue
        firms, subsh = maybe
        # save intermediate file
        pd.DataFrame(data=subsh, index=firms.astype(np.int64)).to_stata(
            "data/intermediate.dta"
        )
        std, cov_std, mal, cov_mal = compute(subsh)
        year = max(year_set)
        if outfile != Path(""):
            post_process(firms, year, std, cov_std, mal, cov_mal, outfile)


def main() -> None:
    logging.basicConfig(
        filename="data/debug.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.info("reading config file")
    verbose = (argv[1] == "--v") if len(argv) > 1 else False
    with open("data/config.json", "r") as config_file:
        config: dict[str, str | int] = load(config_file)

    if verbose:
        print("read config file", config)

    input_file: str = cast(str, config["input_data"])
    outfile = Path(cast(str, config["output_data"]))
    iter_size = cast(int, config["year_iteration"])

    logging.info(f"reading input file {input_file}")
    data: pd.DataFrame = (
        pd.read_stata(input_file)
        if input_file.endswith(".dta")
        else pd.read_csv(input_file, sep="\t")
    )

    if verbose:
        print("read input file successfully")

    data = data[["firm", "nclass", "year"]]
    data["year"] = data["year"].astype(np.uint16)
    # sort by nclass and create a new tclass independant of naming of nclass just in case
    logging.info("replacing nclass by tclass")
    data["firm"] = data.firm.astype(np.uint64)
    data["nclass"] = data.nclass.astype(np.uint32)

    tclass_replacements = dict(
        (k, int(v)) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique()))
    )

    data["tclass"] = data.nclass.replace(tclass_replacements)
    data["year"] = data.year.astype(np.uint16)

    logging.info("sorting years")
    data = data.sort_values("year")

    logging.info("Launchung core computation")
    print(f"Computing with configurations: {input_file=}, {outfile=}, {iter_size=}")
    core(data, iter_size, outfile)

    logging.info("reducing data")
    if verbose:
        print("concatenating data files into one")
    reduce_data(outfile)


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")
