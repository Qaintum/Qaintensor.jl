using Test
using TestSetExtensions
using Qaintensor
using Qaintessent
using LinearAlgebra
using Random
using BenchmarkTools
using TensorOperations

# @testset ExtendedTestSet "open_mps" begin
#
#     T = Tensor(rand(8,8,8))
#     mps = OpenMPS(T, 3)
#
#     gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
#     mps_contract=contract($gmps)
#     mps_svd=contract($mps; er=0.0)
#
#     @test mps_contract ≈ mps_svd
# end
#
# @testset ExtendedTestSet "closed_mps" begin
#
#     T1 = Tensor(rand(2,5))
#     T2 = Tensor(rand(5,2,3))
#     T3 = Tensor(rand(3,2))
#     mps = ClosedMPS([T1, T2, T3])
#
#     gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
#     mps_contract=contract(gmps)
#     mps_svd=contract(mps; er=0.0)
#
#     @test mps_contract ≈ mps_svd
# end
#
# @testset ExtendedTestSet "conversion from state-vector to mps" begin
#     N = 5
#     b1 = randn(ComplexF64, (2))
#     b2 = randn(ComplexF64, (2))
#     b3 = randn(ComplexF64, (2))
#     b4 = randn(ComplexF64, (2))
#     b5 = randn(ComplexF64, (2))
#
#     ψ_ref = kron(b1,b2,b3,b4,b5)
#
#     mps = MPS(ψ_ref)
#
#     ψ = contract(mps)
#
#     @test ψ_ref ≈ reshape(ψ, (2^N,))
# end
#
# @testset ExtendedTestSet "test permute MPS" begin
#     N = 6
#     b = AbstractArray{ComplexF64}[]
#     for i in 1:N
#         push!(b, randn(ComplexF64, (2)))
#     end
#
#     order = randperm(N)
#
#     ψ_ref = kron(b...)
#     ψ_perm = kron(b[order]...)
#
#     mps = MPS(ψ_ref)
#     permute!(mps, order)
#
#     ψ = contract(mps)
#     @test ψ_perm ≈ reshape(ψ, (2^N,))
# end
#
# @testset ExtendedTestSet "test permute MPS exceptions" begin
#
#     N = 6
#     b = AbstractArray{ComplexF64}[]
#     for i in 1:N
#         push!(b, randn(ComplexF64, (2)))
#     end
#
#     mps = MPS(kron(b...))
#
#     short_order = [1, 2, 3]
#     @test_throws ErrorException("Given permutation must be same length as number of Tensors in MPS") permute!(mps, short_order)
#
#     long_order = [1, 2, 3, 4, 5, 6, 7]
#     @test_throws ErrorException("Given permutation must be same length as number of Tensors in MPS") permute!(mps, short_order)
#
#     repeat_order = [1, 1, 1, 1, 1, 1]
#     @test_throws ErrorException("Permutation order cannot contain repeat values") permute!(mps, repeat_order)
#
#     negative_order = [1,-2, 3, 4, -5, 6]
#     @test_throws ErrorException("Permutation order can only contain positive values") permute!(mps, negative_order)
#
#     negative_order = [1, 2, 8, 4, 5, 6]
#     @test_throws ErrorException("Wire numbers in permutation order cannot exceed number of wires in MPS") permute!(mps, negative_order)
# end
#
#
# @testset ExtendedTestSet "test switch MPS qubits" begin
#     N = 6
#     b = AbstractArray{ComplexF64}[]
#     for i in 1:N
#         push!(b, randn(ComplexF64, (2)))
#     end
#     order = Int[1, 3, 2, 5, 6, 4]
#     ψ_ref = kron(b...)
#     ψ_perm = kron(b[order]...)
#
#     mps = MPS(ψ_ref)
#     switch!(mps, [(2,3), (4,6), (4,5)])
#
#     ψ = contract(mps)
#     @test ψ_perm ≈ reshape(ψ, (2^N,))
#
#     mps = MPS(ψ_ref)
#     switch!(mps, 2, 3)
#     switch!(mps, 4, 6)
#     switch!(mps, 4, 5)
#
#     ψ = contract(mps)
#     @test ψ_perm ≈ reshape(ψ, (2^N,))
# end
#
# @testset ExtendedTestSet "test switch MPS exceptions" begin
#
#     N = 6
#     b = AbstractArray{ComplexF64}[]
#     for i in 1:N
#         push!(b, randn(ComplexF64, (2)))
#     end
#
#     mps = MPS(kron(b...))
#
#     @test_throws ErrorException("Wire indices `i` and `j` must be positive") switch!(mps, 1, -2)
#     @test_throws ErrorException("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS") switch!(mps, 7, 4)
#
#     @test_throws ErrorException("Wire indices `i` and `j` must be positive") switch!(mps, [(3, 4), (1, -2)])
#     @test_throws ErrorException("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS") switch!(mps, [(9, 10), (1, -2)])
# end
#
# @testset ExtendedTestSet "test switch neighboring MPS qubits" begin
#     N = 6
#     b = AbstractArray{ComplexF64}[]
#     for i in 1:N
#         push!(b, randn(ComplexF64, (2)))
#     end
#     order = Int[1, 3, 4, 5, 6, 2]
#     ψ_ref = kron(b...)
#     ψ_perm = kron(b[order]...)
#
#     mps = MPS(ψ_ref)
#     switch!(mps, 4)
#     switch!(mps, 3)
#     switch!(mps, 2)
#     switch!(mps, 1)
#
#     ψ = contract(mps)
#     @test ψ_perm ≈ reshape(ψ, (2^N,))
# end
#
# @testset ExtendedTestSet "test switch neighboring MPS qubits exceptions" begin
#     N = 6
#     b = AbstractArray{ComplexF64}[]
#     for i in 1:N
#         push!(b, randn(ComplexF64, (2)))
#     end
#     ψ_ref = kron(b...)
#
#     mps = MPS(ψ_ref)
#
#     @test_throws BoundsError switch!(mps, 7)
#     @test_throws BoundsError switch!(mps, -2)
# end


@testset ExtendedTestSet "closed_mps" begin
    N = 4
    T1 = Tensor(rand(2,3))
    T2 = Tensor(rand(3,2,4))
    T3 = Tensor(rand(4,2,5))
    # T4 = Tensor(rand(5,2,4))
    T5 = Tensor(rand(5,2))
    # mps = ClosedMPS([T1,T2,T3,T4,T5])
    mps = ClosedMPS([T1,T2,T3,T5])
    mps2 = deepcopy(mps)
    # switch!(mps2, 1, 2)
    # ψ2 = contract(mps2)
    # ψ = contract(mps)
    # println(ψ2)
    # println(ψ)
    gmps = GeneralTensorNetwork(deepcopy(mps.tensors), deepcopy(mps.contractions), deepcopy(mps.openidx))
    #
    cg1 = controlled_circuit_gate(4, 1, YGate(), N)
    cg2 = controlled_circuit_gate(4, 1, YGate(), N)
    #
    # # cg1 = single_qubit_circuit_gate(i, XGate(), N)
    # # cg2 = single_qubit_circuit_gate(i, XGate(), N)
    M = 2
    d = 2

    cg_matrix = reshape(Qaintessent.matrix(cg1.gate), fill(d, 2*M)...)

    # ref = [t.data for t in gmps.tensors]
    # ψ_ref = ncon(ref, [[-1, 1], [1, -2, 2], [2, -3]])
    # @test ψ_ref ≈ contract(mps)
    #
    # tensors = decompose(cg_matrix, M)
    # contraction_indices = [[-1, -2, 4],[4, -3, -4, 5], [5, -6, -7]]
    # println(ncon(tensors, contraction_indices))
    # a = vcat(ref, tensors)

    # contraction_indices = [[-1, 1], [1, 5, 2], [2, 6], [5, -2, 4], [4, 6, -3]]
    # contraction_indices = [[7, 1], [1, 8, 2], [2, 9], [7, -2, 3], [3, 8, -4, 4], [4, 9, -6]]

    # ψ_ans = ncon(a, contraction_indices)
    # println(ψ_ans)

    tensor_circuit!(mps, cg1)
    tensor_circuit!(gmps, cg2)
    # switch!(mps, 1,2)
    # println("Running contraction")
    ψ = contract(gmps)
    println(ψ)
    # @test ψ ≈ ψ_ans

    ψ_ans = contract(mps)
    println(ψ_ans)
    @test ψ ≈ ψ_ans
end
