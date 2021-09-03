"""
Parsy is a program that computes various distances between companies' carateristics
"""
from __future__ import annotations

from json import load
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd

from sparsy.core import process
from sparsy.utils import clean_up


def main() -> None:
    with open("data/config.json", "r") as config_file:
        config: dict[str, str | int] = load(config_file)

    input_file: str = cast(str, config["input_data"])
    data: pd.DataFrame = pd.read_csv(
        input_file,
        sep="\t",
        usecols=["firm", "nclass", "year"],
        dtype={"firm": np.uint32, "nclass": "category", "year": np.uint16},
    )
    process(data, config)
    clean_up()


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")
