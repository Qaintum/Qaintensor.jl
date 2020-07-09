
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
            #@assert 1 ≤ j.second ≤ ndims(net.tensors[j.first])
            (1 ≤ j.second ≤ ndims(net.tensors[j.first])) || 
                throw(DimensionMismatch("attempt to contract dimension $(j.second) of $(ndims(net.tensors[j.first]))-dimensional tensor $(j.first)"))
            indexlist[j.first][j.second] = i
        end
        
        # Check dimensions match
        for j in 1:length(ts.idx)-1
            t1 = ts.idx[j]
            t2 = ts.idx[j+1]
            size1 = size(net.tensors[t1.first].data, t1.second)
            size2 = size(net.tensors[t2.first].data, t2.second)
            (size1 == size2) || 
                throw(DimensionMismatch("dimension $(t1.second) of tensor $(t1.first) doesn't match dimension $(t2.second) of tensor $(t2.first)"))
        end

    end
    for (i, oi) in enumerate(net.openidx)
        # tensor leg must not participate in a contraction
        #@assert indexlist[oi.first][oi.second] == 0
        (indexlist[oi.first][oi.second] == 0) || 
            throw(DimensionMismatch("dimension $(oi.second) of tensor $(oi.first) is supposed to be open, but it is already contracted"))
        # last qubit corresponds to fastest varying index
        indexlist[oi.first][oi.second] = i - length(net.openidx) - 1
    end

    # consistency check
    for (i, idx) in enumerate(indexlist), (j, k) in enumerate(idx)
        #@assert i != 0
        (k != 0) || throw(DimensionMismatch("dimension $j of tensor $i is neither open nor contracted"))
    end

    # for now simply forward the contraction operation to 'ncon'
    return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
end
