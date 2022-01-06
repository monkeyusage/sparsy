module Sparsy

import JSON, StatFiles, CSV
using DataFrames
using ProgressBars
using FreqTables

function get_replacements(classes::Vector{T})::Dict{T, UInt64} where {T<:Number}
    # map unique elements to ints
    replacements = Dict{T, UInt64}()
    for (tclass, nclass) in enumerate(unique(classes))
        replacements[nclass] = tclass
    end
    replacements
end

function tclass_corr(matrix::Matrix)::Matrix
    var = matrix'matrix
    base_var = copy(var)
    s = size(var)[1]
    for i in 1:s
        for j in 1:s
            if var[i,i] == 0 || var[j,j] == 0
                continue
            end
            var[i,j] = var[i, j] / (sqrt(base_var[i,i]) * sqrt(base_var[j,j]))
        end
    end
    var
end

function dot_zero(matrix::Matrix{<:Number})::Array{Float32}
    K = I = size(matrix)[1]
    J = size(matrix)[2]

    out = Array{Float32}(undef, K)

    Threads.@threads for k in 1:K
        total = Float32(0)
        for i in 1:I
            if i == k continue end
            for j in 1:J
                @inbounds total = total + (matrix[k, j] * matrix[i, j])
            end
        end
        out[k] = total
    end
    out
end

function mahalanobis(biggie::Matrix{T}, small::Matrix{T})::Vector{Float32} where {T<:Number}
    K = size(biggie)[1]
    J = size(biggie)[2]
    I = size(small)[2]

    out = Array{Float32}(undef, K)

    Threads.@threads for k in 1:K
        total = Float32(0)
        for i in 1:I
            if i == k continue end
            for j in 1:J
                @inbounds total = total + (biggie[k, j] * small[j, i])
            end
        end
        out[k] = total
    end
    out
end

function compute_metrics(matrix::Matrix)::NTuple{4, Vector{Float32}}
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

function main()
    # dataprep
    config = JSON.parsefile("data/config.json")
    input_file = config["input_data"]
    output_file = config["output_data"]
    iter_size = config["year_iteration"]

    data = DataFrame(StatFiles.load(input_file))
    data = data[:, ["year", "firm", "nclass"]]
    data[!, "year"] = map(UInt16, data[!, "year"])
    data[!, "nclass"] = map(UInt32, data[!, "nclass"])

    # replace nclass by tclass and save mapping to json
    replacements = get_replacements(data[!, "nclass"])
    open("data/replacements.json", "w") do f
        write(f, JSON.json(replacements, 4))
    end

    replace!(data[!, "nclass"], replacements...) # replace nclass to be tclass
    rename!(data, "nclass" => "tclass") # rename nclass to tclass
    sort!(data, "year") # sort by year

    CSV.write("data/tmp/intermediate.csv", data)
    years = [year for year in data[!, "year"][1]:data[!, "year"][end]]

    # main loop
    for year_set in ProgressBar(Iterators.partition(years, iter_size))
        sub_df = filter(:year => in(Set(year_set)), data)
        year = min(year_set...)
        freq = freqtable(sub_df, :firm, :tclass)
        firms, tclasses = names(freq) # extract index from NamedMatrix
        freq = convert(Matrix{eltype(freq)}, freq) # just keep the data
        CSV.write(
            "data/tmp/intermediate_$year.csv",
            Tables.table(freq, header=tclasses)
        )
        std, cov_std, mal, cov_mal = compute_metrics(freq)
        CSV.write(
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

    # merge all output files together
    files = []
    for file in readdir("data/tmp/")
        if !endswith(file, "_tmp.csv") continue end
        push!(files, CSV.read("data/tmp/$file", DataFrame))
    end

    CSV.write(output_file, vcat(files...))
end

export main

end