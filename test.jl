using CUDA
using Statistics
using BenchmarkTools

multiplier = 100

gpu = CUDA.rand(Float32, 1024 * multiplier, 16)
cpu = copyto!(zeros(Float32, 1024 * multiplier, 16), gpu)
w_gpu = CUDA.ones(Float32, 1024 * multiplier)
w_cpu = ones(Float32, 1024 * multiplier)

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
            for j in 1:M
                if index == i continue end
                out[index] += matrix[index, j] * matrix[i, j] * weights[index]
            end
        end
    end
    return nothing
end

function dot_zero(
    matrix::CuArray{Float32, 2, CUDA.Mem.DeviceBuffer},
    weights::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}
)::Array{Float32}
    len = length(matrix)
    N, M = size(matrix)
    threads_per_block = 256
    blocks = Int(ceil(N / threads_per_block))

    # blocks, threads = size(matrix)
    out = CUDA.zeros(Float32, N)

    @cuda threads=threads_per_block blocks=blocks dot_zero_gpu_kernel!(matrix, weights, out, len, N, M)
    return Array(out)
end

# compile those
cpu_res = dot_zero(cpu, w_cpu)
gpu_res = dot_zero(gpu, w_gpu)

@assert mean(abs.(gpu_res - cpu_res)) < 1

# benchmark it
println("using multiplier value : ", multiplier)
@btime dot_zero(gpu, w_gpu)
@btime dot_zero(cpu, w_cpu)