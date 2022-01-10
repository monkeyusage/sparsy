import Pkg

libraries = [
    "JSON",
    "DataFrames",
    "StatFiles",
    "CSV",
    "BenchmarkTools",
    "ProgressBars",
    "FreqTables",
    "Debugger"
]

for pkg in libraries
    Pkg.add(pkg)
end

import JSON:json


function main()::Nothing
    mkpath("data/tmp")
    open("data/config.json", "w") do f
        config = Dict(
            "input_data" => "data/data.dta",
            "weight_data" => "data/weight.dta",
            "output_data"=> "data/output.dta",
            "year_iteration" => 3
        )
        write(f, json(config, 4))
    end
    return nothing
end
