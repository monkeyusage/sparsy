from __future__ import annotations

from os import listdir, remove
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

import pandas as pd

T = TypeVar("T")
def chunker(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    l = len(seq)
    if size < 0:
        yield seq
        return
    for idx in range(0, l, size):
        yield seq[idx:min(idx + size, l)]


def clean_up(out_file: Path) -> None:
    # if an out_df.tsv already exists we delete it and replace it
    # we remove the temp files once we have replace the out_df file
    if out_file.exists():
        # remove old out file
        remove(out_file)

    # concatenate all the spill files and save them as out_df.tsv
    dfs: list[pd.DataFrame] = []

    for file in listdir(str(out_file.parent)):
        if not file.endswith("tmp.tsv"):
            continue
        df = pd.read_csv(f"{out_file.parent}/{file}", sep="\t")
        dfs.append(df)
    out_df = pd.concat(dfs, ignore_index=True)
    out_df = out_df.sort_values("firm")
    out_df.to_csv(out_file, sep="\t", index=False)

    # delete the spill files
    for file in listdir(f"{out_file.parent}"):
        if file.endswith("tmp.tsv"):
            remove(f"{out_file.parent}/{file}")
