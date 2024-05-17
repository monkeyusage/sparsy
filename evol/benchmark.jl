using DataFrames
using BenchmarkTools
using CSV


function dot_zero(matrix::Array{Float32, 2})::Array{Float32}
    """
    # vectorized version of the following operations with M (n, m) => NM (n, n) => n
    out = matrix * matrix' => creates a matrix we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    N, M = size(matrix)
    out = Array{Float32, 1}(undef, N)
    
    Threads.@threads for i in 1:N
        total = zero(Float32)
        unweighted_total = zero(Float32)
        remainder = zero(Float32)
        @inbounds for ii in 1:N
            if (i == ii) continue end
            @inbounds for j in 1:M
                total += matrix[i, j] * matrix[ii, j]
            end
        end
        @inbounds out[i] = total
    end

    return out
end


function main()
  df = DataFrame(CSV.File("data/sample.tsv", delim='\t'))[1:50_000, :]
  data = Matrix{Float32}(df)
  dot_zero(data[1:10, :]) # jit compile
  @time dot_zero(data)
end

main()
