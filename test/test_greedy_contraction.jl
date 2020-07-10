# using Test
# using TestSetExtensions
# using Qaintensor
# using Qaintessent
# using BenchmarkTools
# using LinearAlgebra
# using StatsBase: sample
# using TensorOperations
#
# function shift_summation(S::Summation, step::Integer)
#    return Summation([S.idx[i].first + step => S.idx[i].second for i in 1:2])
# end
#
# function shift_pair(P::Pair{Integer, Integer}, step::Integer)
#     return P.first + step => P.second
# end
#
# function rand_U(M)
#     U, _ = qr(rand(ComplexF64, 2^M, 2^M))
#     @assert U*adjoint(U) ≈ I
#     return Array(U)
# end
#
#
# function log_depth_mps_TN(M, N, C)
#     Nlayers = C*Int(round(log2(N)))
#     gates = AbstractCircuitGate{N}[]
#     for j in 1:Nlayers
#         itergates = (j%2 == 1) ? (1:M:N-M+1) : (M:M:N-M+1)
#         for i in itergates
#             U = rand_U(M)
#             g = CircuitGate{M, N, AbstractGate{M}}(Tuple(collect(i:i+M-1)), MatrixGate(U))
#             push!(gates, g)
#         end
#     end
#     cgc = CircuitGateChain{N}(gates)
#     # make tensor network
#
#     T = TensorNetwork([], [], [])
#     # oldbond = abs(rand(Int, 1)[1] % 10) + 2
#     oldbond = 5
#     push!(T.tensors, Tensor(randn(ComplexF64, (2, oldbond))))
#     for i in 1:N-2
#         # newbond = abs(rand(Int, 1)[1] % 10)  + 2
#         newbond = 5
#         push!(T.tensors, Tensor(randn(ComplexF64, (2, oldbond, newbond))))
#         oldbond = newbond
#     end
#     push!(T.tensors, Tensor(randn(ComplexF64, (2, oldbond))))
#
#     # contract virtual legs
#     push!(T.contractions, Summation([1=>2, 2=>2]))
#     for i in 2:N-1
#         push!(T.contractions, Summation([i=>3, i+1=>2]))
#     end
#
#     for i in 1:N
#         push!(T.openidx, i=>1)
#     end
#
#     tensor_circuit!(T, cgc)
#     T_prime = TensorNetwork(copy(T.tensors), copy(T.contractions), copy(T.openidx) )
#
#     l = length(T.tensors)
#     step = length(T_prime.tensors)
#     for i in 1:l
#         push!(T_prime.tensors, conj(T.tensors[i]))
#     end
#
#     for i in 1:length(T.openidx)
#         push!(T_prime.contractions, Summation([T_prime.openidx[i], shift_pair(T.openidx[i], step)]))
#     end
#
#     for (i, con) in enumerate(T.contractions)
#         push!(T_prime.contractions, shift_summation(con, step))
#     end
#
#     T_prime.openidx=Pair[]
#
#     T, cgc
# end
#
#
# @testset ExtendedTestSet "greedy contraction" begin
#     M = 2
#     N = 4
#     ψ, cgc = log_depth_mps_TN(M, N, 1);
#     size_dict, rep, openidx = tn_to_ssa(ψ)
# end
