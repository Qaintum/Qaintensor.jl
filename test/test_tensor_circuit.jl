using Test
using TestSetExtensions
using Qaintensor
using Qaintessent
using Qaintmodels

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

@testset ExtendedTestSet "decompose 2-qubit" begin
    # Test decomposition of 2-qubit gate
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

    cgc = [
        circuit_gate((3), X, (1)),
        circuit_gate((3), Y, (1)),
        circuit_gate((1), Y, (2)),
        circuit_gate((2), Z, (1))
            ]

    ψref = apply(cgc, contract(ψ)[:])

    tensor_circuit!(ψ, cgc; is_decompose=true)

    @test ψref ≈ contract(ψ)[:]
end
#
@testset ExtendedTestSet "decompose 3-qubit" begin
    # Test decomposition of 3-qubit gate
    N = 4
    # initial MPS wavefunction
    ψ = GeneralTensorNetwork([], [], [])
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 6))))
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 6, 7))))
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 7, 5))))
    push!(ψ.tensors, Tensor(randn(ComplexF64, (2, 5))))
    # contract virtual legs
    push!(ψ.contractions, Summation([1=>2, 2=>2]))
    push!(ψ.contractions, Summation([2=>3, 3=>2]))
    push!(ψ.contractions, Summation([3=>3, 4=>2]))

    # open (physical) legs
    push!(ψ.openidx, 1=>1)
    push!(ψ.openidx, 2=>1)
    push!(ψ.openidx, 3=>1)
    push!(ψ.openidx, 4=>1)

    cgc = [
    circuit_gate((4,1), SwapGate(), (3)),
    circuit_gate(2, YGate()),
    circuit_gate((3), XGate(), (1)),
    circuit_gate(2, YGate()),
    circuit_gate(4, X, (1,2)),
    circuit_gate(3, ZGate()),
    ]
    ψref = apply(cgc, contract(ψ)[:])

    tensor_circuit!(ψ, cgc; is_decompose=true)

    @test ψref ≈ contract(ψ)[:]
end
