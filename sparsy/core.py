from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm

from sparsy.numeric import compute
from sparsy.utils import chunker


def process(data: pd.DataFrame, iter_size: int, outfile: Path, IO: bool = True) -> None:
    # sort by nclass and create a new tclass independant of naming of nclass just in case
    tclass_replacements = dict(
        (k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique()))
    )
    data["tclass"] = data.nclass.replace(tclass_replacements)

    # iterate through n_sized chunks
    data = data.sort_values("year")
    years: list[int] = list(range(data["year"].min(), data["year"].max() + 1))

    for year_set in tqdm(chunker(years, iter_size)):
        data_chunk = data[data["year"].isin(set(year_set))]
        if data_chunk.empty:
            continue
        # crosstab on firm and class
        year = max(year_set)

        i, firms = pd.factorize(data_chunk["firm"])
        j, _ = pd.factorize(data_chunk["tclass"])
        ij, tups = pd.factorize(list(zip(i, j)))
        subsh = csr_matrix((np.bincount(ij), tuple(zip(*tups))))

        std, cov_std, mal, cov_mal = compute(subsh)

        if IO:
            # df creation for further saving
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
