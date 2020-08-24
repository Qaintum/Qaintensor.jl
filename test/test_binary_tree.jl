using Test
using TestSetExtensions
using Qaintensor

@testset ExtendedTestSet "binary_tree" begin

    for N in [2, 4, 8]
        ψ = rand(ComplexF64, 2^N)
        ψ_tree = binary_tree(ψ);
        @test reshape(contract(ψ_tree), 2^N) ≈ ψ
    end
end
