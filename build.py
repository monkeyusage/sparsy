from os import system

system("python -m isort sparsy .")
system("python -m black sparsy *.py")
