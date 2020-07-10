using Test
using TestSetExtensions
using Qaintensor


@testset ExtendedTestSet "tensor circuit" begin

    N = 3

    # initial MPS wavefunction
    ψ = GeneralTensorNetwork([], [], [])
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 6))))
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 6, 7))))
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 7))))
    # contract virtual legs
    push!(ψ.contractions, Summation([1=>2, 2=>2]))
    push!(ψ.contractions, Summation([2=>3, 3=>2]))
    # open (physical) legs
    push!(ψ.openidx, 1=>1)
    push!(ψ.openidx, 2=>1)
    push!(ψ.openidx, 3=>1)

    cgc = qft_circuit(N)

    # conventional statevector representation, as reference
    ψref = apply(cgc, contract(ψ)[:])

    # using tensor network representation
    tensor_circuit!(ψ, cgc)

    @test ψref ≈ contract(ψ)[:]

end
