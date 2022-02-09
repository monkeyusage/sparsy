using CUDA
using LinearAlgebra

gpu = CUDA.rand(Float32, 10, 4)
cpu = copyto!(zeros(Float32, 10, 4), gpu)

function dot_zero(matrix::Array{Float32, 2})::Matrix{Float32}
    out = matrix * matrix' # => creates a matrix we cannot store in RAM
    out[diagind(out)] .= 0
    out = sum(out, dims=2)
    out
end

dot_zero(cpu)