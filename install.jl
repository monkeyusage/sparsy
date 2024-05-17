import Pkg

libraries = [
    "JSON",
    "DataFrames",
    "StatFiles",
    "CSV",
    "BenchmarkTools",
    "ProgressBars",
    "FreqTables",
    "Debugger",
    "CUDA",
    "StaticArrays"
]

for pkg in libraries
    Pkg.add(pkg)
end

import JSON:json


function main()
    mkpath("data/tmp")
    config = json(Dict(
        "input_data" => "data/data.dta",
        "weight_data" => "data/weight.dta",
        "output_data"=> "data/output.csv",
        "year_iteration" => 1
     ), 4)

    open("data/config.json", "w") do f
        write(f, config)
    end
end

main()

