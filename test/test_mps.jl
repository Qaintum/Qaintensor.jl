using Test
using TestSetExtensions
using Qaintensor
using LinearAlgebra

@testset ExtendedTestSet "open_mps" begin

    T = Tensor(rand(2,2,2))
    mps = OpenMPS(T, 3)

    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract=contract(gmps)
    mps_svd=contract(mps; er=0.0)

    @test mps_contract ≈ mps_svd.data
end

@testset ExtendedTestSet "closed_mps" begin

    T1 = Tensor(rand(2,5))
    T2 = Tensor(rand(5,2,3))
    T3 = Tensor(rand(3,2))
    mps = ClosedMPS([T1, T2, T3])

    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract=contract(gmps)
    mps_svd=contract(mps; er=0.0)

    @test mps_contract ≈ mps_svd.data
end

@testset ExtendedTestSet "conversion from state-vector to mps" begin
    N = 5
    b1 = randn(ComplexF64, (2))
    b2 = randn(ComplexF64, (2))
    b3 = randn(ComplexF64, (2))
    b4 = randn(ComplexF64, (2))
    b5 = randn(ComplexF64, (2))

    ψ_ref = kron(b1,b2,b3,b4,b5)

    mps = MPS(ψ_ref)

    ψ = contract(mps)

    @test ψ_ref ≈ reshape(ψ.data, (2^N,))
end
