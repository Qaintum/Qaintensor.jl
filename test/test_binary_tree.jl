using Test
using TestSetExtensions
using LinearAlgebra
using Qaintensor

@testset ExtendedTestSet "binary_tree" begin

    for N in [2, 4, 8]
        ψ = rand(ComplexF64, 2^N)
        ψ_tree = binary_tree(ψ);
        @test reshape(contract(ψ_tree), 2^N) ≈ ψ
    end
end

@testset ExtendedTestSet "apply_mpo_binary_tree" begin

    N, M = 4, 2

    iwire = (1,3)

    U = rand(ComplexF64, 2^M ,2^M)
    ψ = rand(ComplexF64, 2^N)

    ψ_tree = binary_tree(ψ);
    ψ_prime = apply_MPO_binarytree(ψ_tree, MPO(U), (1,3));

    ψ_mps = MPS(ψ);
    ψ_mps_prime = apply_MPO(ψ_mps, MPO(U), iwire);

    U = reshape(kron(U, Matrix(1I, 2, 2)), (fill(2, 2*3)...,) );
    U = permutedims(U, [2,1,3,5,4,6])
    U = reshape(U, (2^3, 2^3))
    U = kron(U, Matrix(1I, 2, 2));

    @test reshape(contract(ψ_prime), 2^4) ≈ U*ψ
    @test reshape(contract(ψ_prime), 2^4) ≈ reshape(contract(ψ_mps_prime), 2^4)
end
