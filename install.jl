import Pkg

libraries = [
    "JSON",
    "DataFrames",
    "StatFiles",
    "CSV",
    "BenchmarkTools",
    "ProgressBars"
]

for pkg in libraries
    Pkg.add(pkg)
end