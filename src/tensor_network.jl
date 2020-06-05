
"""
Abstract representation of a single summation within a contraction operation
"""
struct Summation
    # first index refers to tensor, and second index to tensor leg
    idx::AbstractVector{Pair{Integer,Integer}}
end


"""
Tensor network, consisting of tensors and contraction operations
"""
mutable struct TensorNetwork
    # list of tensors
    tensors::AbstractVector{Tensor}
    # contractions, specified as list of summations
    contractions::AbstractVector{Summation}
    # ordered "open" (uncontracted) indices (list of tensor and leg indices)
    openidx::AbstractVector{Pair{Integer,Integer}}
end

function Base.deepcopy(net::TensorNetwork)
    a = TensorNetwork([],[],[])
    a.tensors = deepcopy(net.tensors)
    a.contractions = deepcopy(net.contractions)
    a.openidx = deepcopy(net.openidx)
    return a
end
