from os import system, path, mkdir

print("> Creating virtual environment for python and installing requirements")
system("python -m venv venv")
result = system(r".\venv\Scripts\activate && python -m pip install -r requirements.txt")

if not path.exists("data"):
    print("> did not find a data folder creating it, you should paste your data files there for convenience")
    mkdir("data")
