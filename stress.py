from __future__ import annotations

from itertools import product
from json import load
from time import perf_counter
from os.path import exists

import pandas as pd
from tqdm import tqdm
from memory_profiler import memory_usage

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
                # "range\trows\tfirms\tclasses\telapsed\tmemory\n"
            )

    config: dict[str, int | str] = {"output_data": "data/tmp"}
    parameter_combinations: list[tuple[int, ...]] = list(product(*parameter_dict.values()))
    for parameters in tqdm(parameter_combinations):
        print(f"launching simulation with parameters, {parameters}")
        date_range, n_rows, n_firms, n_classes = parameters
        data = pd.DataFrame(
            data=gen_data(n_rows, n_classes, n_firms),
            columns=["firm", "nclass", "year"],
        )
        config["date_range"] = date_range
        t0 = perf_counter()
        # memory = memory_usage(proc=(process, [data, config]), max_usage=True)
        process(data, config)
        t1 = perf_counter()
        elapsed = t1-t0
        print(f"elapsed_time: {elapsed}")
        
        with open("data/trace.tsv", "a") as trace_file:
            p = '\t'.join(parameters)
            # trace_file.write(f"{p}\t{elapsed}\t{memory}\n")
            trace_file.write(f"{p}\t{elapsed}\n")


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Total elapsed time: {t1-t0}")
