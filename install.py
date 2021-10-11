from json import dump
from os import mkdir, path, system

print("> Creating virtual environment for python and installing requirements")
if not path.exists("venv"):
    system("python -m venv venv")
else:
    system(r".\venv\Scripts\activate && python -m pip uninstall -r requirements.txt -y")

result = system(r".\venv\Scripts\activate && python -m pip install -r requirements.txt")
system(r".\venv\Scripts\activate && python setup.py build_ext --inplace")

if not path.exists("data"):
    print("> did not find a data folder creating it")
    mkdir("data")
    if not path.exists("data/spills"):
        mkdir("data/spills")

config = {
    "input_data": "data/data.dta",
    "output_data": "data/spills/output.dta",
    "year_iteration": 3,
    "n_cores": 0,
    "stress": {
        "iteration_sizes": [3, 5],
        "n_rows": [10000000],
        "n_firms": [500000],
        "n_classes": [6, 500],
    },
}

print("overwriting/creating configurations file")
with open("data/config.json", "w") as config_file:
    dump(config, config_file, indent=4, sort_keys=True)
