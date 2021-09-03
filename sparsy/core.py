from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix

from sparsy.numeric import compute
from sparsy.utils import chunker


def process(data: pd.DataFrame, config: dict[str, str | int], IO:bool=True) -> None:

    # sort by nclass and create a new tclass independant of naming of nclass just in case
    tclass_replacements = dict(
        (k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique()))
    )
    data["tclass"] = data.nclass.replace(tclass_replacements)

    iter_size: int = cast(int, config["iteration_size"])

    out_dir: str = cast(str, config["output_data"])

    # iterate through n_sized chunks
    data = data.sort_values("year")

    for idx, data_chunk in enumerate(chunker(data, iter_size)):
        # crosstab on firm and class
        median_year = data_chunk["year"].median()
        min_year = data_chunk["year"].min()
        max_year = data_chunk["year"].max()

        i, firms = pd.factorize(data_chunk["firm"])
        j, _ = pd.factorize(data_chunk["tclass"])
        ij, tups = pd.factorize(list(zip(i, j)))
        subsh = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
        # cross = pd.crosstab(dataframe["firm"], dataframe["tclass"])
        # subsh = cross.values
        # firms = cross.index.values
        
        std, cov_std, mal, cov_mal = compute(subsh)

        if IO:
            # df creation for further saving
            df = pd.DataFrame(
                {
                    "max_year" : max_year,
                    "min_year": min_year,
                    "median_year": median_year,
                    "firm": firms,
                    "std": std,
                    "cov_std": cov_std,
                    "mal": mal,
                    "cov_mal": cov_mal,
                }
            )
            # saving into memory
            df.to_csv(f"{out_dir}/spill_{idx}.tsv", sep="\t", index=False)
