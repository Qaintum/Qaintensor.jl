
using Test
using TestSetExtensions
using Qaintensor
using Qaintessent
using BenchmarkTools
using LinearAlgebra
using StatsBase: sample
using TensorOperations
using LightGraphs, SimpleWeightedGraphs
using TikzGraphs
using Statistics
#
# @testset ExtendedTestSet "qft contraction optimization circuit" begin
#     N = 4
#     θ = 0.2
#     max_bond_dim = 8
#     cgc = CircuitGateChain{N}([
#         controlled_circuit_gate((1), 3, RzGate(θ + 0.1), N),
#         controlled_circuit_gate((2), 3, RzGate(θ + 0.2), N),
#         controlled_circuit_gate((1), 3, RzGate(θ + 0.3), N),
#         controlled_circuit_gate((1, 3), 2, RzGate(θ + 0.4), N),
#     ])
#     catn_contraction_cost(cgc, max_bond_dim)
# end

@testset ExtendedTestSet "qft contraction optimization circuit" begin
    N = 4
    θ = 0.2
    max_bond_dim = 8
    cgc = CircuitGateChain{N}([
        controlled_circuit_gate((1), 3, RzGate(θ + 0.1), N),
        controlled_circuit_gate((2), 3, RzGate(θ + 0.2), N),
        controlled_circuit_gate((1), 3, RzGate(θ + 0.3), N),
        controlled_circuit_gate((1, 3), 2, RzGate(θ + 0.4), N),
    ])
    swap_contraction_cost(cgc, max_bond_dim)
end
