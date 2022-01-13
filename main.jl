using JSON: json, parsefile
using StatFiles: load as dtaload
using CSV: write as csvwrite, read as csvread
using DataFrames: DataFrame, replace!, rename!, sort!
using ProgressBars: ProgressBar
using FreqTables: freqtable
using Tables: table
using CUDA

function get_replacements(classes::Vector{T})::Dict{T, UInt64} where {T<:Number}
    # map unique elements to ints
    replacements = Dict{T, UInt64}()
    for (tclass, nclass) in enumerate(unique(classes))
        replacements[nclass] = tclass
    end
    replacements
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
    var
end

function dot_zero(matrix::Array{Float32, 2})::Array{Float32}
    """
    # vectorized version of the following operations with M (n, m) => NM (n, n) => n
    out = matrix * matrix' => creates a matrix we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    X, Y = Xp, _ = size(matrix)

    out = Array{Float32, 1}(undef, X)

    Threads.@threads for x in 1:X
        total = zero(Float32)
        @inbounds for xp in 1:Xp
            @inbounds @simd for y in 1:Y
                total = (xp == x) ? total : (total + matrix[x, y] * matrix[xp, y])
            end
        end
        @inbounds out[x] = total
    end
    out
end

function dot_zero_gpu_kernel!(mat::CuArray{Float32, 2}, out::CuArray{Float32, 1})::Nothing
    X, Y = Xp, _ = size(mat)

    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y +threadIdx().y

    if (row < X) && (col < Y)
        @inbounds for x in 0:(X-1)
            total = zero(Float32)
            @inbounds for xp in 0:(Xp-1)
                if xp == x continue end
                @inbounds @simd for y in 0:(Y-1)
                    total += mat[x, y] * mat[xp, y]
                    # total += mat[(row*X) + y] * mat[(xp + y]
                end
            end
            @inbounds out[x] = total
        end
    end

    return nothing
end

function dot_zero(matrix::CuArray{Float32, 2})::CuArray{Float32, 1}
    blocks, threads = size(matrix)
    out = CUDA.zeros(Float32, blocks)

    @cuda threads=threads blocks=blocks dot_zero_gpu_kernel!(matrix, out)
    return out
end

function mahalanobis(biggie::Matrix{T}, small::Matrix{T})::Array{Float32} where {T<:Number}
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
                total += biggie[k, j] * small[j, i]
            end
        end
        @inbounds out[k] = total
    end
    out
end

function compute_metrics(matrix::Matrix)::NTuple{4, Array{Float32}}
    α = (matrix ./ sum(matrix, dims=2))
    # scale down precision to 32 bits / 16 bits breaks
    α = convert(Matrix{Float32}, α)
    
    β = tclass_corr(α)

    ω = α ./ sqrt.(sum(α .* α, dims=2))

    # generate std measures
    std = dot_zero(ω)
    cov_std = dot_zero(α)

    # # generate mahalanobis measure
    ma = mahalanobis(ω, β*ω')
    cov_ma = mahalanobis(α, β*α')
    std, cov_std, ma, cov_ma
end

function dataprep!(data::DataFrame, weights::DataFrame)::NTuple{DataFrame, 2}
    data = data[:, ["year", "firm", "nclass"]]
    data[!, "year"] = map(Int16, data[!, "year"])
    data[!, "nclass"] = map(UInt32, data[!, "nclass"])
    data[!, "firm"] = parse.(Int, data[!, "firm"])

    weights[!, "year"] = map(Int16, weights[!, "year"])
    weights[!, "weight"] = map(Float32, weights[!, "weight"])
    weights[!, "firmid"] = parse.(Int, weights[!, "firmid"])

    # replace nclass by tclass and save mapping to json
    replacements = get_replacements(data[!, "nclass"])
    open("data/replacements.json", "w") do f
        write(f, json(replacements, 4))
    end

    replace!(data[!, "nclass"], replacements...) # replace nclass to be tclass
    rename!(data, "nclass" => "tclass") # rename nclass to tclass
    sort!(data, ["year", "firm"]) # sort by year and firmid

    sort!(weights, ["year", "firmid"])

    data
end


function slice_chop(data::DataFrame, weights::DataFrame, year_set::Array{UInt16})
    sub_df = filter(:year => in(Set(year_set)), data)
    sub_weights = filter(:year => in(Set(year_set)), weights)
    year = max(year_set...)
    freq = freqtable(sub_df, :firm, :tclass)
    firms, tclasses = names(freq) # extract index from NamedMatrix
    freq = convert(Matrix{eltype(freq)}, freq) # just keep the data
    csvwrite(
        "data/tmp/intermediate_$year.csv",
        table(freq, header=tclasses)
    )
    @time std, cov_std, mal, cov_mal = compute_metrics(freq)
    csvwrite(
        "data/tmp/$(year)_tmp.csv",
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

function main()
    config = parsefile("data/config.json")
    input_file = config["input_data"]
    weights_file = config["weight_data"]
    output_file = config["output_data"]
    iter_size = config["year_iteration"]

    data = DataFrame(dtaload(input_file))
    weights = DataFrame(dtaload(weights_file))
    data, weights = dataprep!(data, weights)

    csvwrite("data/tmp/intermediate.csv", data)

    years = [year for year in data[!, "year"][1]:data[!, "year"][end]]
    for year_set in ProgressBar(Iterators.partition(years, iter_size))
        slice_chop(data, weights, year_set)
    end

    # merge all output files together
    files = [csvread("data/tmp/$file", DataFrame) for file in readdir("data/tmp/") if endswith(file, "_tmp.csv")]
    csvwrite(output_file, vcat(files...))
end

main()
