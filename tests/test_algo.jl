include("../src/cpu.jl")
include("../src/gpu.jl")

using LinearAlgebra
using Test

function naive(nums::Matrix{Float32})
    out = nums * nums'
    out[diagind(out)] .= 0
    sum(out, dims=2)
end


matrix = rand(Float32, 10, 10)
weights = ones(Float32, 10)
@test isapprox(naive(matrix),dot_zero(matrix, weights, nothing, 0))

if CUDA.functional()
    cu_mat = CuArray(matrix)
    cu_weights = CuArray(weights)
    @test isapprox(naive(matrix), dot_zero(cu_mat, cu_weights, nothing, 0))
end
