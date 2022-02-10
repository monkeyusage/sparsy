using JSON: json, parsefile
using StatFiles: load as dtaload
using CSV: write as csvwrite, read as csvread
using DataFrames: DataFrame, replace!, rename!, sort!, groupby, combine
using ProgressBars: ProgressBar
using FreqTables: freqtable
using Tables: table
using CUDA: CuArray, @cuda
using Statistics: mean
using Debugger: @bp

function get_replacements(classes::Vector{T})::Dict{T, UInt64} where {T<:Number}
    # map unique elements to ints
    replacements = Dict{T, UInt64}()
    for (tclass, nclass) in enumerate(unique(classes))
        replacements[nclass] = tclass
    end
    return replacements
end

function tclass_corr(matrix::Array{T, 2})::Array{T, 2} where {T<:Number}
    var = matrix'matrix
    base_var = copy(var)
    s = size(var)[1]
    @inbounds @simd for i in 1:s
        @inbounds @simd for j in 1:s 
            var[i, j] = var[i,i] == 0 || var[j,j] == 0 ? 1 : var[i, j] / (sqrt(base_var[i,i]) * sqrt(base_var[j,j]))
        end
    end
    return var
end

function dot_zero(matrix::Array{Float32, 2}, weights::Array{Float32, 1})::Array{Float32}
    """
    # vectorized version of the following operations with M (n, m) => NM (n, n) => n
    out = matrix * matrix' => creates a matrix we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    N, M = Np, _ = size(matrix)

    out = Array{Float32, 1}(undef, N)

    Threads.@threads for x in 1:N # 10
        total = zero(Float32)
        @inbounds for xp in 1:Np # 10
            if (x == xp) continue end
            @inbounds for y in 1:M # 4
                total += matrix[x, y] * matrix[xp, y] * weights[x]
            end
        end
        @inbounds out[x] = total
    end
    return out
end

function dot_zero_gpu_kernel!(
    mat::CuDeviceMatrix{Float32},
    weight::CuDeviceVector{Float32},
    out::CuDeviceVector{Float32}
)::Nothing

    len = length(mat)
    N, M = size(mat)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (index) < len
        for i in 1:N
            for j in 1:M
                if index == i continue end
                out[index] += mat[index, j] * mat[i, j] * weight[index]
            end
        end
    end
    return nothing
end

function dot_zero(matrix::CuDeviceMatrix{Float32}, weight::CuDeviceVector{Float32})::Array{Float32, 1}
    N = size(matrix)[1]
    threads_per_block = 256
    blocks = ceil(N / threads_per_block)

    # blocks, threads = size(matrix)
    out = CUDA.zeros(Float32, blocks)

    @cuda threads=threads_per_block blocks=blocks dot_zero_gpu_kernel!(matrix, weight, out)
    return Array(out)
end

function mahalanobis(biggie::Array{Float32, 2}, small::Array{Float32, 2}, weights::Array{Float32, 1})::Array{Float32}
    """
    # vectorized version of the following operations
    out = biggie * (small * biggie')
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    K = size(biggie)[1]
    J = size(biggie)[2]
    I = size(small)[2]

    out = Array{Float32}(undef, K)

    Threads.@threads for k in 1:K
        total = Float32(0)
        @inbounds for i in 1:I
            if i == k continue end
            @inbounds @simd for j in 1:J
                total += (biggie[k, j] * small[j, i]) * weights[k]
            end
        end
        @inbounds out[k] = total
    end
    return out
end

function compute_metrics(matrix::AbstractArray{Float32, 2}, weight::Array{Float32})::NTuple{4, Array{Float32}}
    α = (matrix ./ sum(matrix, dims=2))
    # scale down precision to 32 bits
    α = convert(Matrix{Float32}, α)
    
    β = tclass_corr(α)

    ω = α ./ sqrt.(sum(α .* α, dims=2))

    # generate std measures
    std = dot_zero(ω, weight)
    cov_std = dot_zero(α, weight)

    # # generate mahalanobis measure
    ma = mahalanobis(ω, β*ω', weight)
    cov_ma = mahalanobis(α, β*α', weight)
    return std, cov_std, ma, cov_ma
end

function dataprep!(data::DataFrame, weights::DataFrame, no_weight::Bool=false)::NTuple{2, DataFrame}
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

    if !no_weight
        sort!(weights, ["year", "firmid"])
        weights[!, "year"] = map(Int16, weights[!, "year"])
        weights[!, "weight"] = map(Float32, weights[!, "weight"])
        weights[!, "firmid"] = parse.(Int, weights[!, "firmid"])
    end

    return data, weights
end


function chop(
    data::DataFrame,
    weights::DataFrame,
    year_set::Set{UInt16},
    no_weight::Bool = false
)::Union{Nothing, Tuple{Array{Float32, 2}, Array{Float32}, Array{Int64}, UInt16}}
    sub_df = filter(:year => in(year_set), data)
    year = max(year_set...)

    freq = freqtable(sub_df, :firm, :tclass)
    firms, _ = names(freq)
    freq = convert(Array{Float32}, freq)

    sub_weights = !no_weight ? filter(:year => ==(year), weights) : DataFrame(:weight => ones(Float32, size(freq)[1]))

    if isempty(sub_df) | isempty(sub_weights)
        return nothing
    end
    weight = sub_weights[!, "weight"]

    sf = size(freq)[1]
    sw = size(weight)[1]
    
    @assert(sf == sw, "matrices shapes do not match $sf & $sw")

    return freq, weight, firms, year
end

function main(args)
    config = parsefile("data/config.json")
    input_file = config["input_data"]
    weights_file = config["weight_data"]
    output_file = config["output_data"]
    iter_size = config["year_iteration"]

    @assert(endswith(output_file, ".csv"), "output file should be a csv")

    data = DataFrame(dtaload(input_file))

    no_weight = ("no-weight" in args) | (weights_file == "")

    weights = !no_weight ? DataFrame(dtaload(weights_file)) : DataFrame(:weight => ones(Float32, size(data)[1]))
    data, weights = dataprep!(data, weights, no_weight)

    csvwrite("data/tmp/intermediate.csv", data)

    year_range = [UInt16(year) for year in min(data[!, "year"]...):max(data[!, "year"]...)]

    # create vector of sets of years
    years::Vector{Set{UInt16}} = []
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
    println("here we go computing stuff")
    for year_set in ProgressBar(years)
        out = chop(data, weights, year_set, no_weight)
        if isnothing(out); continue; end
        freq, weight, firms, year = out
        @time std, cov_std, mal, cov_mal = compute_metrics(freq, weight)
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
