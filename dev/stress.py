"""Use this file to run stress tests"""
from __future__ import annotations

from itertools import product
from json import load
from os.path import exists
from pathlib import Path
from time import perf_counter

import pandas as pd

from sparsy.core import process
from sparsy.numeric import gen_data


def main():
    with open("data/config.json", "r") as f:
        parameter_dict = load(f)["stress"]

    if not exists("data/trace.tsv"):
        print("did not find trace file, creating one")
        with open("data/trace.tsv", "w") as trace_file:
            trace_file.write("itersize\trows\tfirms\tclasses\telapsed\n")

    parameter_combinations: list[tuple[int, ...]] = list(
        product(*parameter_dict.values())
    )
    print(f"Parameter combinations: {parameter_combinations}")
    for parameters in parameter_combinations:
        print(f"launching simulation with parameters, {parameters}")
        iter_size, n_rows, n_firms, n_classes = parameters
        data = pd.DataFrame(
            data=gen_data(n_rows, n_classes, n_firms),
            columns=["firm", "nclass", "year"],
        )
        t0 = perf_counter()
        process(data, iter_size, outfile=Path(""), IO=False)
        t1 = perf_counter()
        elapsed = t1 - t0
        print(f"elapsed_time: {elapsed}")

        with open("data/trace.tsv", "a") as trace_file:
            p = "\t".join(map(lambda x: str(x), parameters))
            trace_file.write(f"{p}\t{elapsed}\n")


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Total elapsed time: {t1-t0}")
