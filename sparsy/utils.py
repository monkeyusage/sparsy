from __future__ import annotations

from os import getpid, listdir, remove
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

import numpy as np
import pandas as pd
from psutil import Process

T = TypeVar("T")


def chunker(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """
    cuts sequence in n sized sequences and stops when len(sub_sequence) < size
    >>> l = list(range(10))
    >>> print(list(chunker(l, 3)))
    >>> [[0, 1, 2], [1, 2, 3], [2, 3, 4], ..., [7, 8, 9]]
    """
    l = len(seq)
    if size < 0:
        yield seq
        return
    for idx in range(0, l):
        out = seq[idx : min(idx + size, l)]
        if len(out) == size:
            yield out
        else:
            return


def extract_type(dtype: str) -> int:
    """
    >>> import numpy as np
    >>> arr = np.random.randn(size=(50,50), dtype=np.uint64)
    >>> extract_type(arr.dtype.name)
    >>> 64
    """
    return int("".join(filter(str.isdigit, dtype)))


def get_memory_usage() -> int:
    """return python process memory usage in megabytes"""
    mb: int = Process(getpid()).memory_info().rss / 1024 ** 2
    return mb


def reduce_data(out_file: Path) -> None:
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
    out_df.to_stata(out_file) if out_file.name.endswith(".dta") else out_df.to_csv(
        out_file, sep="\t", index=False
    )

    # delete the spill files
    for file in listdir(f"{out_file.parent}"):
        if file.endswith("tmp.tsv"):
            remove(f"{out_file.parent}/{file}")
