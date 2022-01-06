import Pkg

libraries = [
    "JSON",
    "DataFrames",
    "StatFiles",
    "CSV",
    "BenchmarkTools",
    "ProgressBars",
    "FreqTables"
]

for pkg in libraries
    Pkg.add(pkg)
end