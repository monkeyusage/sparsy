from __future__ import annotations

from itertools import islice
from json import load
from os import listdir, path, remove
from typing import Iterable, Iterator, TypeVar, cast

import pandas as pd

T = TypeVar("T")


def chunker(seq:pd.DataFrame, size:int) -> Iterator[pd.DataFrame]:
    return (seq.iloc[pos:pos + size] for pos in range(0, len(seq), size))


def clean_up() -> None:
    with open("data/config.json", "r") as config_file:
        config: dict[str, str | int] = load(config_file)

    out_dir: str = cast(str, config["output_data"])

    # if an out_df.tsv already exists we delete it and replace it
    # we remove the temp files once we have replace the out_df file
    if path.exists(f"{out_dir}/out_df.tsv"):
        # remove old out file
        remove(f"{out_dir}/out_df.tsv")
    
    # concatenate all the spill files and save them as out_df.tsv
    dfs: list[pd.DataFrame] = []
    for file in listdir(out_dir):
        df = pd.read_csv(f"{out_dir}/{file}", sep="\t")
        dfs.append(df)
    out_df = pd.concat(dfs, ignore_index=True)
    out_df.to_csv(f"{out_dir}/out_df.tsv", sep="\t", index=False)
    
    # delete the spill files
    for file in listdir(f"{out_dir}"):
        if file.startswith("spill"):
            remove(f"{out_dir}/{file}")
