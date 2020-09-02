using Test
using TestSetExtensions
using Qaintensor
using Qaintessent
using LinearAlgebra
using Combinatorics
using StatsBase: sample

function randn_complex(size)
    return (randn(size)
     + im*randn(size)) / sqrt(2)
end

function random_mps(N)
    mps = []
    for i in 1:N-2
        push!(mps, randn_complex((2, 2,2)))
    end
    pushfirst!(mps, randn_complex((2, 2)))
    push!(mps, randn_complex((2, 2)))

    return ClosedMPS(Tensor.(mps));
end

@testset ExtendedTestSet "mpo" begin
    N = 2
    A = rand(ComplexF64, 2^N ,2^N)
    U, R = qr(A)
    U = Array(U);
    mpo = MPO(U)
    @test length(mpo.tensors) == N

    GU = MatrixGate(U);
    mpo_gate = MPO(GU)
    @test mpo ≈ mpo_gate

end

@testset ExtendedTestSet "extend_MPO" begin
    A = rand(ComplexF64, 2 ,2)
    B = rand(ComplexF64, 2 ,2)
    U1 = kron(kron(A, Matrix(1I, 2,2)), B)
    U2 = kron(A,B)
    @test contract(MPO(U1)) ≈ contract(extend_MPO(U2, (1,3)))
    @test contract(MPO(U1)) ≈ contract(extend_MPO(MPO(U2), (1,3)))
end

@testset ExtendedTestSet "apply_mpo" begin

    #CNOT tests
    Ucnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];
    Ucnot13 = reshape(kron(Ucnot, Matrix(I, 2, 2)), (2, 2, 2, 2, 2, 2))
    Ucnot13 = permutedims(Ucnot13, [2,1,3,5,4,6])
    Ucnot13 = reshape(Ucnot13, (8, 8));
    # create random MPS tensors, using bond dimensions (1, 3, 4, 1)
    mps = [randn_complex((2, 3)), randn_complex((3, 2, 4)), randn_complex((4, 2))]
    ψ_mps = ClosedMPS(Tensor.(mps));
    ψ = reshape(contract(ψ_mps), 2^3);
    ψ_mps_cnot = apply_MPO(ψ_mps, MPO(Ucnot), (1,3));
    @test reshape(contract(ψ_mps_cnot), 8) ≈ Ucnot13*ψ

    #Test for M-qubit gates acting on non-adjacent qubits out of N qubits
    N, M = 7, 3
    A = rand(ComplexF64, 2^M ,2^M);
    U, R = qr(A)
    U = Array(U);
    GU = MatrixGate(U);

    #arbitrary input
    ψ_mps = random_mps(N)
    ψ = reshape(contract(ψ_mps), 2^N);

    #1st case: iwires are sorted
    #apply MPO with input MPO and CircuitGate
    combi = collect(combinations(1:N, M));
    for (s,com) in enumerate(combi)
        iwires_sorted = Tuple(com)
        circuit_GU_sorted = CircuitGate(iwires_sorted,GU,N); #GATE
        ψ_gate_sorted = apply(circuit_GU_sorted, ψ);
        ψ_mpo_sorted = apply_MPO(ψ_mps, MPO(U), iwires_sorted);
        @test reshape(contract(ψ_mpo_sorted), 2^N) ≈ ψ_gate_sorted
        @test contract(ψ_mpo_sorted) ≈ contract(apply_MPO(ψ_mps, circuit_GU_sorted))

        #2nd case: iwires are not sorted.
        #apply_MPO with input AbstractMatrix and CircuitGate
        w = [1:N...]
        iwires = sample(w, M, replace = false)
        circuit_GU = CircuitGate(Tuple(iwires),GU,N)
        ψ_gate = apply(circuit_GU, ψ)
        ψ_mpo = apply_MPO(ψ_mps, U, Tuple(iwires));
        @test reshape(contract(ψ_mpo), 2^N) ≈ ψ_gate
        @test contract(ψ_mpo) ≈ contract(apply_MPO(ψ_mps, circuit_GU))
    end
end

@testset ExtendedTestSet "tests apply_mpo exceptions" begin

    N, M = 5, 3
    A = rand(ComplexF64, 2^M ,2^M);
    U, R = qr(A)
    U = Array(U);
    ψ_mps = random_mps(N)
    GU = MatrixGate(U);

    w_rep = (1, 3, 3)
    w_neg = (1, -2, 4)
    w_max = (1, 3, 6)
    @test_throws ErrorException("Repeated wires are not valid.") apply_MPO(ψ_mps, MPO(U), w_rep)
    @test_throws ErrorException("Wires must be integers between 1 and n (total number of qudits).") apply_MPO(ψ_mps, MPO(U), w_neg)
    @test_throws ErrorException("Wires must be integers between 1 and n (total number of qudits).") apply_MPO(ψ_mps, MPO(U), w_max)

end
