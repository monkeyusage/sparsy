"""
Parsy is a program that computes various distances between companies' carateristics
"""
from __future__ import annotations

import logging
from json import load
from pathlib import Path
from sys import argv, path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd

path.append(".")

from sparsy.core import core
from sparsy.utils import reduce_data


def main() -> None:
    logging.basicConfig(
        filename="data/debug.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.info("reading config file")
    verbose = (argv[1] == "--v") if len(argv) > 1 else False
    with open("data/config.json", "r") as config_file:
        config: dict[str, str | int] = load(config_file)

    if verbose:
        print("read config file", config)

    input_file: str = cast(str, config["input_data"])
    outfile = Path(cast(str, config["output_data"]))
    iter_size = cast(int, config["iteration_size"])
    cores = cast(int, config["n_cores"])

    logging.info(f"reading input file {input_file}")
    data: pd.DataFrame = (
        pd.read_stata(input_file)
        if input_file.endswith(".dta")
        else pd.read_csv(input_file, sep="\t")
    )

    if verbose:
        print("read input file successfully")

    data = data[["firm", "nclass", "year"]]
    data["year"] = data["year"].astype(np.uint16)
    logging.info("Launchung core computation")
    print(
        f"Computing with configurations: {input_file=}, {outfile=}, {iter_size=}, {cores=}"
    )
    core(data, iter_size, outfile, cores)

    logging.info("reducing data")
    if verbose:
        print("concatenating data files into one")
    reduce_data(outfile)


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Elapsed time: {t1 - t0} seconds")
