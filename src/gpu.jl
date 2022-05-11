using CUDA

function tclass_corr(matrix::CuArray{Float32})::CuArray{Float32}
    var = matrix'matrix
    out = copy(var)
    
    N, _ = size(var)
    threads_per_block = N > 128 ? 128 : 1
    blocks = Int(ceil(N / threads_per_block))

    @cuda threads=threads_per_block blocks=blocks tclass_corr_gpu_kernel!(var, out, N)
    return out
end

function tclass_corr_gpu_kernel!(matrix::CuDeviceMatrix{Float32}, out::CuDeviceMatrix{Float32}, N::Int)::Nothing
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if index <= N
        for i in 1:N
            if (out[index,index] == 0) | (out[i,i] == 0)
                out[index, i] = 1
            else
                out[index, i] =  out[index, i] / (sqrt(matrix[index,index]) * sqrt(matrix[i,i]))
            end
        end
    end
    return nothing
end

function dot_zero_gpu_kernel!(
    matrix::CuDeviceMatrix{Float32},
    weights::CuDeviceVector{Float32},
    out::CuDeviceVector{Float32},
    len::Int,
    N::Int,
    M::Int
)::Nothing
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (index) < len
        for i in 1:N
            if index == i continue end
            for j in 1:M
                @inbounds out[index] += matrix[index, j] * matrix[i, j] * weights[i]
            end
        end
    end
    return nothing
end

function dot_zero(
    matrix::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    weights::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer},
    channel::Nothing,
    index::Int
)::Array{Float32}
    # unused arguments are here to still contain same arguments as CPU version so we can call them seamlessly
    len = length(matrix)
    N, M = size(matrix)
    threads_per_block = 256
    blocks = Int(ceil(N / threads_per_block))

    out = CUDA.zeros(Float32, N)

    @cuda threads=threads_per_block blocks=blocks dot_zero_gpu_kernel!(matrix, weights, out, len, N, M)
    return Array(out)
end

function mahalanobis_gpu_kernel!(
    biggie::CuDeviceMatrix{Float32},
    small::CuDeviceMatrix{Float32},
    weights::CuDeviceVector{Float32},
    out::CuDeviceVector{Float32},
    len::Int,
    N::Int,
    M::Int,
)::Nothing
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (index) < len
        for i in 1:N
            if index == i continue end
            for j in 1:M
                @inbounds out[index] += biggie[index, j] * small[j, i] * weights[i]
            end
        end
    end
    return nothing
end

function mahalanobis(
    biggie::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    small::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    weights::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer},
    channel::Nothing,
    index::Int
)::Array{Float32}
    len = length(biggie)
    N, M = size(biggie)
    threads_per_block = 256
    blocks = Int(ceil(N / threads_per_block))

    # blocks, threads = size(matrix)
    out = CUDA.zeros(Float32, N)

    @cuda threads=threads_per_block blocks=blocks mahalanobis_gpu_kernel!(biggie, small, weights, out, len, N, M)
    return Array(out)
end