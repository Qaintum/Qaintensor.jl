
"""
Abstract representation of a single summation within a contraction operation
"""
struct Summation
    # first index refers to tensor, and second index to tensor leg
    idx::AbstractVector{Pair{Integer,Integer}}
end

function Base.:(==)(S1::Summation, S2::Summation)
    S1.idx == S2.idx
end

"""
Abstract struct representing TensorNetwork
"""
abstract type TensorNetwork end

"""
    GeneralTensorNetwork(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{Integer,Integer}})
    
General Tensor network, consisting of tensors and contraction operations
"""
mutable struct GeneralTensorNetwork <: TensorNetwork
    # list of tensors
    tensors::AbstractVector{Tensor}
    # contractions, specified as list of summations
    contractions::AbstractVector{Summation}
    # ordered "open" (uncontracted) indices (list of tensor and leg indices)
    openidx::AbstractVector{Pair{Integer,Integer}}
end

Base.copy(net::GeneralTensorNetwork) = GeneralTensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))
