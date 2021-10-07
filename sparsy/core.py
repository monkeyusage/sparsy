from __future__ import annotations

import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm

from sparsy.numeric import compute
from sparsy.utils import chunker


def preprocess(
    dataframe: pd.DataFrame, sub_years: list[int]
) -> tuple[int, np.ndarray, csr_matrix] | None:
    data_chunk = dataframe[dataframe["year"].isin(set(sub_years))]
    if data_chunk.empty:
        return None
    # crosstab on firm and class
    year: int = max(sub_years)
    firms: np.ndarray
    subsh: csr_matrix

    i, firms = pd.factorize(data_chunk["firm"])
    j, _ = pd.factorize(data_chunk["tclass"])
    ij, tups = pd.factorize(list(zip(i, j)))
    subsh = csr_matrix((np.bincount(ij), tuple(zip(*tups))), dtype=np.float32)
    return year, firms, subsh


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


def process(
    data: pd.DataFrame, year_set: list[int], matrix_iteration: int, outfile: Path
) -> None:
    sub_data = preprocess(data, year_set)
    if sub_data is None:
        return
    year, firms, subsh = sub_data

    stds, cov_stds, mals, cov_mals = [], [], [], []
    i = 0
    while i < subsh.shape[0]:
        std, cov_std, mal, cov_mal = compute(subsh[i : i + matrix_iteration])
        i += matrix_iteration
        if outfile != Path(""):
            stds.append(std)
            cov_stds.append(cov_std)
            mals.append(mal)
            cov_mals.append(cov_mal)

    if outfile != Path(""):
        # df creation for further saving
        stds = np.concatenate(stds)
        cov_stds = np.concatenate(cov_stds)
        mals = np.concatenate(mals)
        cov_mals = np.concatenate(cov_mals)
        post_process(firms, year, stds, cov_stds, mals, cov_mals, outfile)


def core(data: pd.DataFrame, iter_size: int, matrix_iteration:int, outfile: Path, cores: int = 0) -> None:
    # sort by nclass and create a new tclass independant of naming of nclass just in case
    tclass_replacements = dict(
        (k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique()))
    )
    logging.info("replacing nclass by tclass")
    data["tclass"] = data.nclass.replace(tclass_replacements)

    # iterate through n_sized chunks
    logging.info("sorting years")
    data = data.sort_values("year")
    years: list[int] = list(range(data["year"].min(), data["year"].max() + 1))

    if cores == 0:
        logging.info("launching main process on one process")
        for year_set in tqdm(chunker(years, iter_size)):
            logging.info("processing data using year: %s", year_set)
            process(data, year_set, matrix_iteration, outfile)
    else:
        logging.info("launching main process")
        years_sets = list(chunker(years, iter_size))
        process_years = partial(process, data, outfile=outfile, matrix_iteration=matrix_iteration)
        with Pool(cores) as p:
            p.map(process_years, years_sets)
