
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

@testset ExtendedTestSet "contraction cost for catn contraction" begin
    N = 6
    θ = 0.2
    max_bond_dim = 6
    cgc = CircuitGateChain{N}([
        controlled_circuit_gate((1), 3, RzGate(θ + 0.1), N),
        controlled_circuit_gate((2), 5, RzGate(θ + 0.2), N),
        controlled_circuit_gate((5), 4, RzGate(θ + 0.3), N),
        controlled_circuit_gate((1, 4), 2, RzGate(θ + 0.4), N),
        # controlled_circuit_gate((1, 4), 6, RzGate(θ + 0.4), N),
        controlled_circuit_gate((1), 4, RzGate(θ + 0.3), N),
        controlled_circuit_gate((3), 4, RzGate(θ + 0.3), N),
        controlled_circuit_gate((1), 5, RzGate(θ + 0.3), N),
    ])
    cost = catn_contraction_cost(cgc, max_bond_dim)
    println("Cost of CATN contraction: " * string(cost))
    cost = swap_contraction_cost(cgc, max_bond_dim)
    println("Cost of swap contraction: " * string(cost))
end
