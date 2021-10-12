from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sparsy.numeric_python import compute
from sparsy.utils import chunker


def preprocess(
    dataframe: pd.DataFrame, sub_years: list[int]
) -> tuple[np.ndarray, np.ndarray ] | None:
    df = dataframe[dataframe["year"].isin(set(sub_years))]
    if df.empty:
        return None
    # crosstab on firm and class
    firms: np.ndarray
    subsh: np.ndarray

    ct = (
        df
        .groupby(["firm", "tclass"], as_index=False)
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


def process(
    data: pd.DataFrame, year_set: list[int], outfile: Path
) -> None:
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
    import pdb;pdb.set_trace()
    for year_set in tqdm(chunker(years, iter_size)):
        logging.info("processing data using year: %s", year_set)
        maybe = preprocess(data, year_set)
        if maybe is None:
            continue
        firms, subsh = maybe
        # save intermediate file
        pd.DataFrame(data=subsh, index=firms.astype(np.int64)).to_stata("data/intermediate.dta")
        std, cov_std, mal, cov_mal = compute(subsh)
        year = max(year_set)
        if outfile != Path(""):
            post_process(firms, year, std, cov_std, mal, cov_mal, outfile)
