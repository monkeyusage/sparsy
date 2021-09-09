"""
Parsy is a program that computes various distances between companies' carateristics
"""
from __future__ import annotations

from json import load
from pathlib import Path
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
    iter_size = cast(int, config["iteration_size"])
    outfile = Path(cast(str, config["output_data"]))

    data: pd.DataFrame = pd.read_stata(input_file)
    data = data[["firm", "nclass", "year"]]
    data["year"] = data["year"].astype(np.uint16)

    process(data, iter_size, outfile)
    clean_up(outfile)


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")
