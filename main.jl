using JSON: json, parsefile
using StatFiles: load as dtaload
using CSV: write as csvwrite, read as csvread
using DataFrames: DataFrame, replace!, rename!, sort!
using ProgressBars: ProgressBar
using FreqTables: freqtable
using Tables: table

function get_replacements(classes::Vector{T})::Dict{T, UInt64} where {T<:Number}
    # map unique elements to ints
    replacements = Dict{T, UInt64}()
    for (tclass, nclass) in enumerate(unique(classes))
        replacements[nclass] = tclass
    end
    replacements
end

function tclass_corr(matrix::Matrix{<:Number})::Matrix{<:Number}
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
    X = Xp = size(matrix)[1]
    Y = size(matrix)[2]

    out = Array{Float32}(undef, X)

    Threads.@threads for k in 1:X
        total = Float32(0)
        for i in 1:Xp
            if i == k continue end
            for j in 1:Y
                @inbounds total = matrix[k, j] * matrix[i, j] + total
            end
        end
        out[k] = total
    end

    # # vectorized version
    # out = matrix * matrix'
    # out[diagind(out)] .= 0
    # out = sum(out, dims=2)

    # write a version with while loop on unrolled matrix (1D) and get the same results
    # this will help determining the i, j, k indices for GPU kernel creation
    out
end

# function kernel_example!(out, mat)
#     # this is the part we haev to figure out
#     # x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     if x < size(mat)[1]
#         @inbounds out[x] = mat[idx_k, ind_i] * mat[idx_i, idx_j]
#     end
# end

function kernel_matmul(C, A, B)
    """ 
        Compute C = A * B with
            C[m,n] = A[m,p] * B[p,n]
    """    
    tx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ty = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    m, p = size(A)
    _, n = size(B)
    
    Cvalue = 0.0f0
  
    if (tx <= n) && (ty <= m)
      for k = 1:p
        Cvalue += A[(ty-1)*p + k]*B[(k-1)*n + tx]
        #Cvalue += A[(tx-1)*p + k]*B[(k-1)*m + ty]
      end
      # Write the matrix to device memory; each thread writes one element
      C[(ty-1)*n + tx] = Cvalue
    end
    return nothing
end

""" Compute C = A * B """
function kernel_matmul_fast(C, A, B, m, p)
    tx = threadIdx().x

    # Important: The second @cuDynamicSharedMem allocation needs an offset of sizeof(sA), as it uses a single kernel-level buffer
    sA = @cuDynamicSharedMem(Float32, (m,p))
    sB = @cuDynamicSharedMem(Float32, p, sizeof(sA))

    # Initialize shared memory for A
    for j in 1:p
        @inbounds sA[tx, j] = A[tx, j]
    end

    # Initialize shared memory for B
    if tx == 1
    for j in 1:p
        @inbounds sB[j] = B[j]
    end
    end

    # Wait until all threads finish preloading
    sync_threads()

    for j in 1:2000
    Cvalue = 0.0f0

    if tx <= m
        for i = 1:p
        @inbounds Cvalue += sA[tx, i] * sB[i]
        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
        end
        @inbounds C[tx] = Cvalue
        #@cuprintln(C[tx])
    end
    end

    return nothing
end

# function dot_zero_gpu(mat)
#     n = size(mat)[1]
#     out = CUDA.zeros(n)
#     threads = THREADS_PER_BLOCK
#     blocks = ceil(Int64, n/threads)

#     @cuda threads=threads blocks=blocks kernel_example!(out, mat)
# end

function mahalanobis(biggie::Matrix{T}, small::Matrix{T})::Array{Float32} where {T<:Number}
    K = size(biggie)[1]
    J = size(biggie)[2]
    I = size(small)[2]

    out = Array{Float32}(undef, K)

    Threads.@threads for k in 1:K
        total = Float32(0)
        for i in 1:I
            if i == k continue end
            for j in 1:J
                @inbounds total = biggie[k, j] * small[j, i] + total
            end
        end
        out[k] = total
    end
    out

    # # vectorixed version
    # out = biggie * (small * biggie')
    # out[diagind(out)] .= 0
    # out = sum(out, dims=2)
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

function dataprep!(data::DataFrame)::DataFrame
    data = data[:, ["year", "firm", "nclass"]]
    data[!, "year"] = map(UInt16, data[!, "year"])
    data[!, "nclass"] = map(UInt32, data[!, "nclass"])

    # replace nclass by tclass and save mapping to json
    replacements = get_replacements(data[!, "nclass"])
    open("data/replacements.json", "w") do f
        write(f, json(replacements, 4))
    end

    replace!(data[!, "nclass"], replacements...) # replace nclass to be tclass
    rename!(data, "nclass" => "tclass") # rename nclass to tclass
    sort!(data, "year") # sort by year

    data
end


function slice_chop(data, year_set)
    sub_df = filter(:year => in(Set(year_set)), data)
    year = min(year_set...)
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
    output_file = config["output_data"]
    iter_size = config["year_iteration"]

    data = DataFrame(dtaload(input_file))
    data = dataprep!(data)

    csvwrite("data/tmp/intermediate.csv", data)

    years = [year for year in data[!, "year"][1]:data[!, "year"][end]]
    for year_set in ProgressBar(Iterators.partition(years, iter_size))
        slice_chop(data, year_set)
    end

    # merge all output files together
    files = [csvread("data/tmp/$file", DataFrame) for file in readdir("data/tmp/") if endswith(file, "_tmp.csv")]
    csvwrite(output_file, vcat(files...))
end

main()
