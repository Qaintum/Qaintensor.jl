using Test
using TestSetExtensions
using Qaintensor
using Qaintessent
using LinearAlgebra
using Random
using BenchmarkTools
using TensorOperations

@testset ExtendedTestSet "test open MPS construction" begin

    T = Tensor(rand(2,2,2))
    mps = OpenMPS(T, 3)

    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract=contract(gmps)
    mps_svd=contract(mps; er=0.0)

    @test mps_contract ≈ mps_svd
end

@testset ExtendedTestSet "test closed MPS construction" begin

    T1 = Tensor(rand(2,5))
    T2 = Tensor(rand(5,2,3))
    T3 = Tensor(rand(3,2))
    mps = ClosedMPS([T1, T2, T3])

    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract=contract(gmps)
    mps_svd=contract(mps; er=0.0)

    @test mps_contract ≈ mps_svd
end

@testset ExtendedTestSet "test conversion from state-vector to MPS object" begin
    N = 5
    b1 = randn(ComplexF64, (2))
    b2 = randn(ComplexF64, (2))
    b3 = randn(ComplexF64, (2))
    b4 = randn(ComplexF64, (2))
    b5 = randn(ComplexF64, (2))

    ψ_ref = kron(b1,b2,b3,b4,b5)

    mps = MPS(ψ_ref)

    ψ = contract(mps)

    @test ψ_ref ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test permute MPS" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end

    order = randperm(N)

    ψ_ref = kron(b...)

    ψ_perm = kron(b[order]...)

    mps = MPS(ψ_ref)
    permute!(mps, order)

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test permute MPS exceptions" begin

    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end

    mps = MPS(kron(b...))

    short_order = [1, 2, 3]
    @test_throws ErrorException("Given permutation must be same length as number of Tensors in MPS") permute!(mps, short_order)

    long_order = [1, 2, 3, 4, 5, 6, 7]
    @test_throws ErrorException("Given permutation must be same length as number of Tensors in MPS") permute!(mps, short_order)

    repeat_order = [1, 1, 1, 1, 1, 1]
    @test_throws ErrorException("Permutation order cannot contain repeat values") permute!(mps, repeat_order)

    negative_order = [1,-2, 3, 4, -5, 6]
    @test_throws ErrorException("Permutation order can only contain positive values") permute!(mps, negative_order)

    negative_order = [1, 2, 8, 4, 5, 6]
    @test_throws ErrorException("Wire numbers in permutation order cannot exceed number of wires in MPS") permute!(mps, negative_order)
end


@testset ExtendedTestSet "test switch MPS qubits" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end
    order = Int[1, 3, 2, 5, 6, 4]
    ψ_ref = kron(b...)
    ψ_perm = kron(b[order]...)

    mps = MPS(ψ_ref)
    switch!(mps, [(2,3), (4,6), (4,5)])

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))

    mps = MPS(ψ_ref)
    switch!(mps, 2, 3)
    switch!(mps, 4, 6)
    switch!(mps, 4, 5)

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test switch MPS exceptions" begin

    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end

    mps = MPS(kron(b...))

    @test_throws ErrorException("Wire indices `i` and `j` must be positive") switch!(mps, 1, -2)
    @test_throws ErrorException("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS") switch!(mps, 7, 4)

    @test_throws ErrorException("Wire indices `i` and `j` must be positive") switch!(mps, [(3, 4), (1, -2)])
    @test_throws ErrorException("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS") switch!(mps, [(9, 10), (1, -2)])
end

@testset ExtendedTestSet "test switch neighboring MPS qubits" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end
    order = Int[1, 3, 4, 5, 6, 2]
    ψ_ref = kron(b...)
    ψ_perm = kron(b[order]...)

    mps = MPS(ψ_ref)
    switch!(mps, 4)
    switch!(mps, 3)
    switch!(mps, 2)
    switch!(mps, 1)

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test switch neighboring MPS qubits exceptions" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end
    ψ_ref = kron(b...)

    mps = MPS(ψ_ref)

    @test_throws BoundsError switch!(mps, 7)
    @test_throws BoundsError switch!(mps, -2)
end


@testset ExtendedTestSet "test switch contraction method for MPS" begin
    N = 5
    T1 = Tensor(rand(2,3))
    T2 = Tensor(rand(3,2,4))
    T3 = Tensor(rand(4,2,5))
    T4 = Tensor(rand(5,2,8))
    # T5 = Tensor(rand(8,2,6))
    # T6 = Tensor(rand(6,2,4))
    # T7 = Tensor(rand(4,2,3))
    # T8 = Tensor(rand(3,2,5))
    # T9 = Tensor(rand(5,2,4))
    T10 = Tensor(rand(8,2))

    # mps = ClosedMPS([T1,T2,T3,T4,T5,T6,T7,T8,T9,T10])
    mps = ClosedMPS([T1,T2,T3,T4,T10])
    # mps = ClosedMPS([T1,T2,T3,T4,T7])

    gmps = GeneralTensorNetwork(deepcopy(mps.tensors), deepcopy(mps.contractions), deepcopy(mps.openidx))

    cgc1 = CircuitGateChain{N}(
    [
        controlled_circuit_gate((1,3,2), 5, YGate(), N),
        controlled_circuit_gate((5,3,2), 4, YGate(), N),
        controlled_circuit_gate((1,3,5), 4, YGate(), N),
        controlled_circuit_gate((1,5,4), 2, YGate(), N),
        controlled_circuit_gate((4,1,3), 5, YGate(), N),
        controlled_circuit_gate((2,4,1), 5, YGate(), N),
        controlled_circuit_gate((2,1,), 4, XGate(), N),
        # controlled_circuit_gate((4,10,2), 3, YGate(), N),
        controlled_circuit_gate((1,4,3), 2, RyGate(0.2), N),
        controlled_circuit_gate((2,3), 1, HadamardGate(), N),
    ])

    cgc2 = CircuitGateChain{N}(
    [
        controlled_circuit_gate((1), 5, YGate(), N),
        controlled_circuit_gate((2), 4, YGate(), N),
        controlled_circuit_gate((3,2), 4, YGate(), N),
        controlled_circuit_gate((4), 2, YGate(), N),
        controlled_circuit_gate((2), 5, YGate(), N),
        controlled_circuit_gate((1), 5, YGate(), N),
        controlled_circuit_gate((2), 3, XGate(), N),
        controlled_circuit_gate((4,1,2), 3, YGate(), N),
        controlled_circuit_gate((1,4,3), 2, RyGate(0.2), N),
        controlled_circuit_gate((2,3), 1, HadamardGate(), N),
    ])

    # @btime tensor_circuit!($gmps, $cgc2)
    @btime tensor_circuit!($gmps, $cgc2)
    tensor_circuit!(gmps, cgc2)
    # println(size.(gmps.tensors))
    # @btime contract($gmps)
    @btime contract($gmps)
    ψ = contract(gmps)

    # @btime tensor_circuit!($mps, $cgc2; max_bond_dim=10)
    @btime tensor_circuit!($mps, $cgc2; max_bond_dim=10)
    tensor_circuit!(mps, cgc2; max_bond_dim=10)
    # println(size.(mps.tensors))
    # @btime contract($mps)
    @btime contract($mps)
    ψ_ans = contract(mps)

    @test ψ ≈ ψ_ans  rtol=1e-5
end
