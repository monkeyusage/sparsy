# Sparsy

Hey I though we needed sparse matrices !?
The original problem was that the algorithm needed way too much memory. It was not solved using sparse matrices...

## Installation
- To install the tools simply `git clone https://github.com/monkeyusage/sparsy` then `git pull` and finally run : `julia install.jl` into your terminal

- The only file you should interact with is:
    - `data/config.json` to change input and output data as well as number of years to use by chunk

Before you run the project please pay attention to the configuration file in the data folder

Mention the file names in their dedicated sections. This software takes in dta files and outputs a csv file.

## Options
If you want to process the whole data in one pass simply insert a negative number in the `year_iteration` configuation. `year_iteration` corresponds to the number of years to include each iteration.

If you do not want to use a weights data file then either set the `weight_data` field to "" in the config file or provide the argument `no-weight` when you call the script.
If you have a GPU with CUDA and for some reason want to execute the program on the CPU then provide the `no-gpu` argument when calling the program.

If you want to log intermediate result (WARNING this defeats the purpose of the project in terms of memory performance, only use for debugging), simply add the `use-logger` argument.

For this project we optimised for memory size. This means the code could theoretically run faster but it would require bigger memory size. Even 128gigs of RAM might not be enough for certain jobs so our optimisation is the only viable option.

Choosing smaller year_iteration will likely increase the execution speed. If you have a GPU the code should run significantly faster then on CPU.

## Running the project
Here is how you launch the script: `julia -t {number_of_cpu_cores} main.jl`

If you want to provide additional command line arguments they should go after `main.jl` in you script call

That could be:
    - `julia -t {number_of_cpu_cores} main.jl no-weight no-gpu`

The *default behaviour* of the program is to use the weight file if there is one mentioned in the config and the GPU if your computer has CUDA.
