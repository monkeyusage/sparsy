function tclass_corr(matrix::Array{Float32, 2})::Array{Float32, 2}
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
    N, M = size(matrix)

    out = Array{Float32, 1}(undef, N)

    Threads.@threads for i in 1:N
        total = zero(Float32)
        @inbounds for ii in 1:N
            if (i == ii) continue end
            @inbounds @simd for j in 1:M
                total += matrix[i, j] * matrix[ii, j] * weights[ii]
            end
        end
        @inbounds out[i] = total
    end
    return out
end

function mahalanobis(biggie::Array{Float32, 2}, small::Array{Float32, 2}, weights::Array{Float32, 1})::Array{Float32}
    """
    # vectorized version of the following operations
    out = biggie * (small * biggie')
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    """
    N, M = size(biggie)

    out = Array{Float32}(undef, N)

    Threads.@threads for i in 1:N
        total = Float32(0)
        @inbounds for ii in 1:N
            if ii == i continue end
            @inbounds @simd for j in 1:M
                total += biggie[i, j] * small[j, ii] * weights[ii]
            end
        end
        @inbounds out[i] = total
    end
    return out
end