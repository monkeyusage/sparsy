from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import bsr_matrix

from sparsy.numeric import compute
from sparsy.utils import chunked_iterable


def process(data: pd.DataFrame, config: dict[str, str | int]) -> None:

    # sort by nclass and create a new tclass independant of naming of nclass just in case
    tclass_replacements = dict(
        (k, v) for k, v in zip(data.nclass.unique(), range(data.nclass.nunique()))
    )
    data["tclass"] = data.nclass.replace(tclass_replacements)

    years_groups: np.ndarray = data["year"].unique()
    years_groups.sort()

    size: int = cast(int, config["date_range"])
    size = size if size > 0 else len(years_groups) + 1

    out_dir: str = cast(str, config["output_data"])

    # iterate through year chunks
    for years in chunked_iterable(years_groups, size):
        dataframe = data[data["year"].isin(years)]

        # crosstab on firm and class
        i, firms = pd.factorize(dataframe["firm"])
        j, _ = pd.factorize(dataframe["tclass"])
        ij, tups = pd.factorize(list(zip(i, j)))
        subsh = bsr_matrix((np.bincount(ij), tuple(zip(*tups))))

        year: int = max(years)
        
        std, cov_std, mal, cov_mal = compute(
            matrix=subsh, tech=data["nclass"].nunique()
        )

        # df creation for further saving
        df = pd.DataFrame(
            {
                "year": year,
                "firm": firms,
                "std": std,
                "cov_std": cov_std,
                "mal": mal,
                "cov_mal": cov_mal,
            }
        )
        # saving into memory
        df.to_csv(f"{out_dir}/spill_{year}.tsv", sep="\t", index=False)
