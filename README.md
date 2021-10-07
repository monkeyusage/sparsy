# Sparsy

Hey I though we needed sparse matrices !?
We don't use it here but someday we might have to so let's keep the name that way.

- To install the tools simply `git clone https://github.com/monkeyusage/sparsy` then `git pull` and finally run : `python install.py` into your terminal
- The only files you should interact with are:
    - `data/config.json` to change input and output data as well as number of years to use by chunk, you can also change the number cpu cores for parallel computing
    - `run.bat` to run the project

Before you run the project please pay attention to the configuration file in the data folder

Mention the file names in their dedicated sections. This software takes in dta files and outputs a dta file.

Pay attention to indicate an output folder that is specific to your data output. The script will create temporary files, concatenate them, save the output and delete the temp files. If the software crashes during execution, for any reason, you might end up with lots of files in your directory and cleaning them by hand could get tedious. So just use a specific folder for the output however you might call the file.

If you want to process the whole data in one pass simply insert a negative number in the `year_iteration` configuation. `year_iteration` corresponds to the number of years to include each iteration. You can also split each matrix into multiple submatrices in order to relieve the computer from `n**2` memory usage. Set this number in the `matrix_iteration` value in the config.json.

For this project we optimised for memory size. This means the code could run faster but it would require bigger mem size.
Choosing bigger itersize will not likely increase the execution speed. Quite the converse.

To speed up the process you can chose a number of cores. However, when choosing too many cores or simply having iter_sizes set too big you will run into memory issues if your dataset is too big. Leave `cores=0` for a sequential process