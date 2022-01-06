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