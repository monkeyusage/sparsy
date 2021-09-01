"""
Parsy is a program that computes various distances between companies' carateristics
"""
from __future__ import annotations

from json import load
from os import listdir, path, remove
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd

from sparsy.numeric import compute, gen_data
from sparsy.utils import chunked_iterable

def main() -> None:
    with open("data/config.json", "r") as config_file:
        config: dict[str, str | int] = load(config_file)

    input_file: str = cast(str, config["input_data"])
    data: pd.DataFrame = (
        pd.read_csv(
            input_file,
            sep="\t",
            usecols=["firm", "nclass", "year"],
            dtype={"firm": np.uint32, "nclass": "category", "year": np.uint16},
        )
        if not config["stress_test"]
        else pd.DataFrame(data=gen_data(), columns=("firm", "nclass", "year"))
    )

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
        subsh = pd.crosstab(dataframe["firm"], dataframe["tclass"]).astype(np.float32)
        firms = subsh.index.values.copy()
        year: int = max(years)

        std, cov_std, mal, cov_mal = compute(
            matrix=subsh.values, tech=data["nclass"].nunique()
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

    # concatenate all the spill files
    dfs: list[pd.DataFrame] = []
    for file in listdir(out_dir):
        df = pd.read_csv(f"{out_dir}/{file}", sep="\t")
        dfs.append(df)

    out_df = pd.concat(dfs, ignore_index=True)
    out_df.to_csv(f"{out_dir}/out_df.tsv", sep="\t", index=False)

    # if we're sure that the concatenated file was created successfully we remove the temp files
    if path.exists(f"{out_dir}/out_df.tsv"):
        for file in listdir(f"{out_dir}"):
            if file.startswith("spill"):
                remove(f"{out_dir}/{file}")


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")
