from __future__ import annotations

from itertools import product
from json import load
from time import perf_counter
from os.path import exists

import pandas as pd
from atexit import register
from traceback import print_exc

from sparsy.core import process
from sparsy.numeric import gen_data


def main():
    with open("data/config.json", "r") as f:
        parameter_dict = load(f)["stress"]

    if not exists("data/trace.tsv"):
        print("did not find trace file, creating one")
        with open("data/trace.tsv", "w") as trace_file:
            trace_file.write(
                "range\trows\tfirms\tclasses\telapsed\n"
            )

    config: dict[str, int | str] = {"output_data": "data/tmp"}
    parameter_combinations: list[tuple[int, ...]] = list(product(*parameter_dict.values()))
    print(f"Parameter combinations: {parameter_combinations}")
    for parameters in parameter_combinations:
        print(f"launching simulation with parameters, {parameters}")
        iter_size, n_rows, n_firms, n_classes = parameters
        data = pd.DataFrame(
            data=gen_data(n_rows, n_classes, n_firms),
            columns=["firm", "nclass", "year"],
        )
        config["iteration_size"] = iter_size
        t0 = perf_counter()
        process(data, config, IO=False)
        t1 = perf_counter()
        elapsed = t1-t0
        print(f"elapsed_time: {elapsed}")
        
        with open("data/trace.tsv", "a") as trace_file:
            p = '\t'.join(map(lambda x: str(x),parameters))
            trace_file.write(f"{p}\t{elapsed}\n")

if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Total elapsed time: {t1-t0}")
