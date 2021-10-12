from os import system

system("python -m isort sparsy install.py")
system("python -m black sparsy install.py")
system("python -m mypy sparsy/main.py sparsy/utils.py install.py")
