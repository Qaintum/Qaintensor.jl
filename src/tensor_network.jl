
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


function contract(net::TensorNetwork)

    # TODO: approximate contraction using SVD splittings

    indexlist = [fill(0, ndims(t)) for t in net.tensors]
    for (i, ts) in enumerate(net.contractions)
        for j in ts.idx
            @assert 1 ≤ j.second ≤ ndims(net.tensors[j.first])
            indexlist[j.first][j.second] = i
        end
    end
    for (i, oi) in enumerate(net.openidx)
        # tensor leg must not participate in a contraction
        @assert indexlist[oi.first][oi.second] == 0
        # last qubit corresponds to fastest varying index
        indexlist[oi.first][oi.second] = i - length(net.openidx) - 1
    end
    # consistency check
    for idx in indexlist, i in idx
        @assert i != 0
    end

    # for now simply forward the contraction operation to 'ncon'
    return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
end
