
"""
    Summation

Abstract representation of a single summation within a contraction operation
"""
struct Summation
    # first index refers to tensor, and second index to tensor leg
    idx::Vector{Pair{Integer,Integer}}
end

function Base.:(==)(S1::Summation, S2::Summation)
    S1.idx == S2.idx
end

"""
Abstract struct representing TensorNetwork
"""
abstract type TensorNetwork end

"""
    GeneralTensorNetwork  <: TensorNetwork

General Tensor network, consisting of tensors and contraction operations
"""
mutable struct GeneralTensorNetwork <: TensorNetwork
    # list of tensors
    tensors::Vector{Tensor}
    # contractions, specified as list of summations
    contractions::Vector{Summation}
    # ordered "open" (uncontracted) indices (list of tensor and leg indices)
    openidx::Vector{Pair{Integer,Integer}}
end

Base.copy(net::GeneralTensorNetwork) = GeneralTensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))
