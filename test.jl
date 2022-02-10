using CUDA
using LinearAlgebra

gpu = CUDA.rand(Float32, 1024, 16)
cpu = copyto!(zeros(Float32, 1024, 16), gpu)

function dot_zero_cpu(matrix::Array{Float32, 2})::Matrix{Float32}
    out = matrix * matrix' # => creates a matrix we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    out
end

function dot_zero_gpu_kernel!(mat, out)
    len = length(mat)
    N, M = size(mat)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (index) < len
        for i in 1:N
            for j in 1:M
                if index == i continue end
                out[index] += mat[index, j] * mat[i, j]
            end
        end
    end
    return nothing
end

function dot_zero_gpu(matrix)
    N = size(matrix)[1]
    threads_per_block = 256
    blocks = Int(ceil(N / threads_per_block))

    # blocks, threads = size(matrix)
    out = CUDA.zeros(Float32, N)

    @cuda threads=threads_per_block blocks=blocks dot_zero_gpu_kernel!(matrix, out)
    return Array(out)
end

println(dot_zero_cpu(cpu))
println(dot_zero_gpu(gpu))