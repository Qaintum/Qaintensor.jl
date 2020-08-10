using Test
using TestSetExtensions
using Qaintensor
using Qaintessent
using LinearAlgebra
using Combinatorics
using StatsBase: sample

@testset ExtendedTestSet "mpo" begin

    function randn_complex(size)
        return (randn(size)
         + im*randn(size)) / sqrt(2)
    end

    Ucnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];
    Ucnot13 = reshape(kron(Ucnot, Matrix(I, 2, 2)), (2, 2, 2, 2, 2, 2))
    Ucnot13 = permutedims(Ucnot13, [2,1,3,5,4,6])
    Ucnot13 = reshape(Ucnot13, (8, 8));

    #Test for CNOT
    # create random MPS tensors, using bond dimensions (1, 3, 4, 1)
    mps = [randn_complex((2, 3)), randn_complex((3, 2, 4)), randn_complex((4, 2))]
    ψ_mps = ClosedMPS(Tensor.(mps));
    ψ = reshape(contract(ψ_mps), 2^3);
    ψ_mps_cnot = apply_MPO(ψ_mps, MPO(Ucnot), (1,3));
    @test reshape(contract(ψ_mps_cnot), 8) ≈ Ucnot13*ψ

    #Test for Circuit_mpo
    A = rand(ComplexF64, 2 ,2)
    B = rand(ComplexF64, 2 ,2)
    U1 = kron(kron(A, Matrix(1I, 2,2)), B)
    U2 = kron(A,B)
    @test contract(MPO(U1)) ≈ contract(circuit_MPO(U2, (1,3)))

    #Test for M-qubit gates acting on non-adjacent qubits out of N
    N, M = 7, 3
    combi = collect(combinations(1:N, M));
    for (s,com) in enumerate(combi)
        iwires_sorted = Tuple(com)

        w = [1:N...]
        iwires = sample(w, M, replace = false)

        A = rand(ComplexF64, 2^M ,2^M);
        U, R = qr(A)
        U = Array(U);
        GU = MatrixGate(U);

        circuit_GU_sorted = CircuitGate(iwires_sorted,GU,N); #GATE
        circuit_GU = CircuitGate(Tuple(iwires),GU,N)

        #ARBITRARY INPUT STATE
        mps = []
        for i in 1:N-2
        push!(mps, randn_complex((2, 2,2)))
        end
        pushfirst!(mps, randn_complex((2, 2)))
        push!(mps, randn_complex((2, 2)))

        ψ_mps = ClosedMPS(Tensor.(mps));
        ψ = reshape(contract(ψ_mps), 2^N);

        ψ_gate_sorted = apply(circuit_GU_sorted, ψ);
        ψ_gate = apply(circuit_GU, ψ);
        ψ_mpo_sorted = apply_MPO(ψ_mps, MPO(U), iwires_sorted);
        ψ_mpo = apply_MPO(ψ_mps, U, Tuple(iwires));
        @test reshape(contract(ψ_mpo), 2^N) ≈ ψ_gate
        @test reshape(contract(ψ_mpo_sorted), 2^N) ≈ ψ_gate_sorted
    end

end
