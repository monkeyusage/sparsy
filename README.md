# Sparsy

We needed to multiply rectangular matrices (N,M) together resulting in (N,N) matrices and OutOfMemory errors.

Here is the full naive algorithm in python/numpy:

```python
    def sparsy(matrix:np.ndarray) -> np.ndarray:
        result = matrix.dot(matrix.T)
        np.fill_diagonal(result, 0)
        return result.sum(axis=1)
```

It was not solved using sparse matrices... We simply unrolled the loop trying everything at our disposal to make it fast as possible.
Checking the git history you'll find numba, cython and other tricks but we settled with julia for its type system, ease of use with GPU and overall speed. 

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
    - `julia -t {number_of_cpu_cores} main.jl no-weight no-gpu use-logger`

The *default behaviour* of the program is to use the weight file if there is one mentioned in the config and the GPU if your computer has CUDA.

It will *not* use the logger by default as its use is very computationally expensive, only activate it when you absolutly need it.


### The logger
Logging things can be quite tedious if we do not want to run out of memory. We create a buffered Channel, you can think of it as a pipe with N slots allocated to it. The pipe is filled by the main thread and consumed by a task that writes the data to local files. We tried saving all the data in one datastructure but it resulted in crashes to memory errors. The Pipe is buffered meaning that if the writer does not go fast enough the producer will wait for the writer to free space on the channel to continue its work. The problem is that we cannot have multiple writers on the same file simultaneously, its ends up in race conditions.

To sum up the logger is to be used with caution. If you want to alter its code get familiar with channels `https://docs.julialang.org/en/v1/manual/asynchronous-programming/#Communicating-with-Channels`. 
