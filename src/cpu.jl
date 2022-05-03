LogDict = Union{Dict{NTuple{2, Integer}, Vector{AbstractFloat}}, Nothing}
function log_value!(dict::LogDict, value::AbstractFloat)::Nothing
    if !isnothing(dict)
        if haskey(dict, (i, ii))
            push!(dict[(i, ii)], value)
        else
            dict[(i, ii)] = [value]
        end
    end
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
    return
end


function dot_zero(
    matrix::Array{Float32, 2},
    weights::Array{Float32, 1},
    logger_dict::LogDict
)::Tuple{Array{Float32}, LogDict}
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
            @inbounds for j in 1:M
                # logging firm pair and the value associated to it
                value = matrix[i, j] * matrix[ii, j]
                log_value!(logger_dict, value)
                total += value * weights[ii]
            end
        end
        @inbounds out[i] = total
    end

    return out, logger_dict
end

function mahalanobis(
    biggie::Array{Float32, 2},
    small::Array{Float32, 2},
    weights::Array{Float32, 1},
    logger_dict::LogDict
)::Tuple{Array{Float32}, LogDict}
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
            @inbounds for j in 1:M
                # logging firm pair and the value associated to it
                value = biggie[i, j] * small[j, ii]
                log_value!(logger_dict, value)
                total += value * weights[ii]
            end
        end
        @inbounds out[i] = total
    end
    return out, logger_dict
end