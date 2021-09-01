from os import system, path, mkdir
from json import dump

print("> Creating virtual environment for python and installing requirements")
if not path.exists("venv"):
    system("python -m venv venv")
else:
    system(r".\venv\Scripts\activate && python -m pip uninstall -r requirements.txt")

result = system(r".\venv\Scripts\activate && python -m pip install -r requirements.txt")

if not path.exists("data"):
    print("> did not find a data folder creating it")
    mkdir("data")
    
    config = {
        "input_data":"",
        "output_data":"",
        "date_range":3
    }

    with open("data/config.json", "w") as config_file:
        dump(config, config_file)