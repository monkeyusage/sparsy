include("../src/cpu.jl")
include("../src/gpu.jl")

using LinearAlgebra
using Test

function naive_dot(nums::Matrix{Float32})
    out = nums * nums'
    out[diagind(out)] .= 0
    sum(out, dims=2)
end

function naive_mal(nums::Matrix{Float32}, other::Matrix{Float32})
    out = nums * other
    out[diagind(out)] .= 0
    sum(out, dims=2)
end

matrix = rand(Float32, 10, 10)
small = rand(Float32, 3, 10)
weights = ones(Float32, 10)

α = (matrix ./ sum(matrix, dims=2))
β = tclass_corr(α)
ω = α ./ sqrt.(sum(α .* α, dims=2))

@testset "DotZero" begin
    @test isapprox(naive_dot(ω),dot_zero(ω, weights, nothing, 0))

    if CUDA.functional()
        cu_mat = CuArray(ω)
        cu_weights = CuArray(weights)
        @test isapprox(naive_dot(matrix), dot_zero(cu_mat, cu_weights, nothing, 0))
    end
end

@testset "Mahalanobis" begin
    @test isapprox(naive_mal(ω, β*ω'),mahalanobis(ω, β*ω', weights, nothing, 0))

    if CUDA.functional()
        cu_mat = CuArray(ω)
        cu_beta = CuArray(β)
        cu_weights = CuArray(weights)
        @test isapprox(naive_mal(ω, β*ω'), mahalanobis(cu_mat, cu_beta*cu_mat',cu_weights, nothing, 0))
    end
end