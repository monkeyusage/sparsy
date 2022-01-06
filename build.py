from os import system

system("python -m isort pysparsy install.py")
system("python -m black pysparsy install.py")
system("python -m mypy pysparsy/main.py pysparsy/utils.py install.py")
