struct LogMessage
    key::NTuple{2, Int}
    index::Int
    value::Float32
end

function tclass_corr(matrix::Array{Float32, 2})::Array{Float32, 2}
    """
    # correlation over N * M matrix that return M * M correlation matrix
    this one does not use threading since M is usually a small numner
    """
    var = matrix'matrix
    base_var = copy(var)
    s = size(var)[1] # arrays are 1 indexed
    @inbounds @simd for i in 1:s
        @inbounds @simd for j in 1:s 
            var[i, j] = var[i,i] == 0 || var[j,j] == 0 ? 1 : var[i, j] / (sqrt(base_var[i,i]) * sqrt(base_var[j,j]))
        end
    end
    return var
end


function dot_zero(
    matrix::Array{Float32, 2},
    weights::Array{Float32, 1},
    channel::Union{Nothing, Channel{LogMessage}},
    index::Int
)::Array{Float32}
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
        remainder = zero(Float32)
        @inbounds for ii in 1:N
            if (i == ii) continue end
            @inbounds for j in 1:M
                value = matrix[i, j] * matrix[ii, j]
                total += value * weights[ii]
            end
            if !isnothing(channel) # put LogMessage into the channel, this will block if the buffer is full
                msg = LogMessage((i, ii), index, (total - remainder))
                put!(channel, msg) # remove the remainder to extract value attributed to ii
                remainder = total
            end
        end
        @inbounds out[i] = total
    end

    return out
end

function mahalanobis(
    biggie::Array{Float32, 2},
    small::Array{Float32, 2},
    weights::Array{Float32, 1},
    channel::Union{Nothing, Channel{LogMessage}},
    index::Int
)::Array{Float32}
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
        remainder = Float32(0)
        @inbounds for ii in 1:N
            if ii == i continue end
            @inbounds for j in 1:M
                # logging firm pair and the value associated to it
                value = biggie[i, j] * small[j, ii]
                total += value * weights[ii]
            end
            if !isnothing(channel) # put LogMessage into the channel, this will block if the buffer is full
                msg = LogMessage((i, ii), index, (total - remainder))
                put!(channel, msg) # remove the remainder to extract value attributed to ii
                remainder = total
            end
        end
        @inbounds out[i] = total
    end

    return out
end
