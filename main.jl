using JSON: json, parsefile
using StatFiles: load as dtaload
using CSV: write as csvwrite, read as csvread
using DataFrames: DataFrame, replace!, rename!, sort!, groupby, combine
using ProgressBars: ProgressBar
using FreqTables: freqtable
using Tables: table
using Statistics: mean
using CUDA

include("gpu.jl")
include("cpu.jl")

function get_replacements(classes::Vector{T})::Dict{T, UInt64} where {T<:Number}
    # map unique elements to ints
    replacements = Dict{T, UInt64}()
    for (tclass, nclass) in enumerate(unique(classes))
        replacements[nclass] = tclass
    end
    return replacements
end

function dataprep!(data::DataFrame, weights::DataFrame, use_weight::Bool=true)::NTuple{2, DataFrame}
    data = data[:, ["year", "firm", "nclass"]]
    data[!, "year"] = map(Int16, data[!, "year"])
    data[!, "nclass"] = map(UInt32, data[!, "nclass"])
    data[!, "firm"] = parse.(Int, data[!, "firm"])

    # replace nclass by tclass and save mapping to json
    replacements = get_replacements(data[!, "nclass"])
    open("data/replacements.json", "w") do f
        write(f, json(replacements, 4))
    end

    replace!(data[!, "nclass"], replacements...) # replace nclass to be tclass
    rename!(data, "nclass" => "tclass") # rename nclass to tclass
    sort!(data, ["year", "firm"]) # sort by year and firmid

    if use_weight
        weights[!, "year"] = map(Int16, weights[!, "year"])
        weights[!, "weight"] = map(Float32, weights[!, "weight"])
        weights[!, "firmid"] = parse.(Int, weights[!, "firmid"])
        sort!(weights, ["year", "firmid"])
    end

    return data, weights
end


function slice(
    data::DataFrame,
    weights::DataFrame,
    year_set::Set{UInt16},
    use_weight::Bool = true,
    use_gpu::Bool = true
)::Union{Nothing, Tuple{AbstractArray{Float32, 2}, AbstractArray{Float32}, Array{Int64}, UInt16}}
    sub_df = filter(:year => in(year_set), data)
    year = max(year_set...)

    if isempty(sub_df) return nothing end

    freq = freqtable(sub_df, :firm, :tclass)
    firms, _ = names(freq)

    freq = use_gpu ? convert(CuArray{Float32}, freq) : convert(Array{Float32}, freq)

    if length(freq) < 2 return nothing end

    weight = use_weight ? filter(row -> row.year==year, weights)[!, "weight"] : ones(Float32, size(freq)[1])
    weight = use_gpu ? convert(CuArray{Float32}, weight) : weight

    sf = size(freq)[1]
    sw = size(weight)[1]
    
    @assert(sf == sw, "matrices shapes do not match $sf & $sw")

    return freq, weight, firms, year
end

function compute_metrics(matrix::AbstractArray{Float32, 2}, weight::AbstractArray{Float32})::NTuple{4, Array{Float32}}
    α = (matrix ./ sum(matrix, dims=2))
    
    # compute matrix of correlations between classes (m x m)
    β = tclass_corr(α)

    # normalize the values inside matrix
    # sum(α .* α, dims=2) == (α*α')[diagind(α)]
    ω = α ./ sqrt.(sum(α .* α, dims=2))

    # generate std measures
    std = dot_zero(ω, weight)
    cov_std = dot_zero(α, weight)

    # # generate mahalanobis measure
    ma = mahalanobis(ω, β*ω', weight)
    cov_ma = mahalanobis(α, β*α', weight)
    return map((x) -> 100 * x, (std, cov_std, ma, cov_ma))
end

function main(args)
    config = parsefile("data/config.json")
    input_file = config["input_data"]
    weights_file = config["weight_data"]
    output_file = config["output_data"]
    iter_size = config["year_iteration"]

    @assert(endswith(output_file, ".csv"), "output file should be a csv")

    data = DataFrame(dtaload(input_file))

    use_weight =  (weights_file != "") & !("no-weight" in args)

    weights = use_weight ? DataFrame(dtaload(weights_file)) : DataFrame(:weight => ones(Float32, size(data)[1]))
    data, weights = dataprep!(data, weights, use_weight)

    csvwrite("data/tmp/intermediate.csv", data)

    year_range = [UInt16(year) for year in min(data[!, "year"]...):max(data[!, "year"]...)]

    # create vector of sets of years
    years = []
    if iter_size > 0
        for i in eachindex(year_range)
            if i + iter_size - 1 < length(year_range)
                push!(years, Set(year_range[i:i+iter_size-1]))
            else
                push!(years, Set(year_range[i:end]))
            end
        end
    else push!(years, Set(year_range))
    end

    # compute metrics for each set of years
    println("Starting computation using $(length(years)) batches")
    if CUDA.functional()
        println("CUDA is available, you should get a significant speed up compared to the CPU version")
        println("If you still want to disable GPU usage use the no-gpu command line argument when launching the script")
    end

    use_gpu = CUDA.functional() & !("no-gpu" in args)
    if !CUDA.functional(); println("GPU not available"); end
    if (CUDA.functional() & !use_gpu); println("GPU available but ignored, computation might take a while"); end
    if use_gpu; println("CUDA available, using GPU"); end

    for year_set in ProgressBar(years)
        out = slice(data, weights, year_set, use_weight, use_gpu)
        if isnothing(out); continue; end
        freq, weight, firms, year = out
        std, cov_std, mal, cov_mal = compute_metrics(freq, weight)
        csvwrite("data/tmp/$(year)_tmp.csv",
            DataFrame(
                "std" => std,
                "cov_std" => cov_std,
                "mal" => mal,
                "cov_mal" => cov_mal,
                "firm" => firms,
                "year" => year
            )
        )
    end

    # merge all output files together, delete tmp files
    files = [csvread("data/tmp/$file", DataFrame) for file in readdir("data/tmp/") if endswith(file, "_tmp.csv")]
    csvwrite(output_file, vcat(files...))

    for file in readdir("data/tmp")
        if endswith(file, "_tmp.csv")
            rm("data/tmp/$file")
        end
    end
end

main(ARGS)
