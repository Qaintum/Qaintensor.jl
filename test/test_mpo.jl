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
    @testset  "mpo constructions" begin
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

    @testset  "extend_MPO self-consistency" begin
        A = rand(ComplexF64, 2 ,2)
        B = rand(ComplexF64, 2 ,2)
        U1 = kron(kron(A, Matrix(1I, 2,2)), B)
        U2 = kron(A,B)
        @test contract(MPO(U1)) ≈ contract(extend_MPO(U2, (3,1)))
        @test contract(MPO(U1)) ≈ contract(extend_MPO(MPO(U2), (3,1)))
    end

    @testset  "extend_MPO CG check " begin
        N = 3
        A = rand(ComplexF64, 2 ,2)
        QA, R = qr(A)
        B = rand(ComplexF64, 2 ,2)
        QB, R = qr(B)

        U = kron(QA, QB)
        GU = circuit_gate((3, 1), MatrixGate(U))

        @test reshape(Qaintessent.sparse_matrix(GU), (fill(2, 2N)...)) ≈ contract(extend_MPO(U, (3,1)))
        @test reshape(Qaintessent.sparse_matrix(GU), (fill(2, 2N)...)) ≈ contract(extend_MPO(MPO(U), (3,1)))
    end

    @testset  "apply_mpo" begin
        @testset "apply_mpo CNOT" begin
            N = 5

            b1 = randn(ComplexF64, (2))
            b2 = randn(ComplexF64, (2))
            b3 = randn(ComplexF64, (2))
            b4 = randn(ComplexF64, (2))
            b5 = randn(ComplexF64, (2))

            cntrl = rand(1:N)
            targ = rand(1:N-1)

            if targ >= cntrl
                targ += 1
            end

            GU = circuit_gate(targ, X, cntrl)
            ψ = kron(b1,b2,b3,b4,b5)

            Ucnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];
            mpo_gate = MPO(Ucnot)
            mps = MPS(ψ)

            @test contract(apply_MPO(mps, mpo_gate, (targ, cntrl)))[:] ≈ apply(ψ, GU)
        end

        @testset "apply_mpo multi-qubit gate" begin
            #Test for M-qubit gates acting on non-adjacent qubits out of N qubits
            N, M = 8, 4
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
                circuit_GU_sorted = CircuitGate(iwires_sorted,GU); #GATE
                ψ_gate_sorted = apply(ψ, circuit_GU_sorted);
                ψ_mpo_sorted = apply_MPO(ψ_mps, MPO(U), iwires_sorted);
                @test reshape(contract(ψ_mpo_sorted), 2^N) ≈ ψ_gate_sorted
                @test contract(ψ_mpo_sorted) ≈ contract(apply_MPO(ψ_mps, circuit_GU_sorted))

                #2nd case: iwires are not sorted.
                #apply_MPO with input AbstractMatrix and CircuitGate
                w = [1:N...]
                iwires = sample(w, M, replace = false)
                circuit_GU = CircuitGate(Tuple(iwires),GU)
                ψ_gate = apply(ψ, circuit_GU)
                ψ_mpo = apply_MPO(ψ_mps, U, Tuple(iwires));
                @test reshape(contract(ψ_mpo), 2^N) ≈ ψ_gate
                @test contract(ψ_mpo) ≈ contract(apply_MPO(ψ_mps, circuit_GU))
            end
        end
    end

    @testset  "tests apply_mpo exceptions" begin

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
end
