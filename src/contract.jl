
using BenchmarkTools

"""
    contract(net)

Fully contract a given TensorNetwork object.
"""

function contract(net::TensorNetwork; optimize=false)
    indexlist, legcosts, openidx = tn_to_ssa(net)

    # TODO: approximate contraction using SVD splittings
    if optimize
        # printstyled("  time for opt_einsum optimization: ", color=:blue)
        sequence, cost = optimal_contraction_order(net, legcosts, indexlist)
        for i in 1:length(indexlist)
            for j in 1:length(indexlist[i])
                if indexlist[i][j] > 0
                    indexlist[i][j] = findall(x->x==indexlist[i][j], sequence)[1]
                end
            end
        end
        # @btime TensorOperations.ncon([t.data for t in $net.tensors], $indexlist)
        return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
    end
    # for now simply forward the contraction operation to 'ncon'
    # @btime TensorOperations.ncon([t.data for t in $net.tensors], $indexlist)
    return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
end
