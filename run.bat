@echo off
rem replace the file name by whatever file you want, must be a .dta or .tsv
set file=data\data.tsv
venv\Scripts\activate && python main.py %file%