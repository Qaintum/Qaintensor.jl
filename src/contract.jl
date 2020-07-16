# """
#     contraction_rep(net::TensorNetwork)
# Switch tensors to get SSA representation for contraction optimization
# """
# function contract_rep(net::TensorNetwork)
#
#     leg_costs = Dict{Int, Int}()
#     indexlist = [fill(0, ndims(t)) for t in net.tensors]
#     for (i, ts) in enumerate(net.contractions)
#         for j in ts.idx
#             @assert 1 ≤ j.second ≤ ndims(net.tensors[j.first])
#             indexlist[j.first][j.second] = i
#             leg_costs[i] = size(net.tensors[j.first])[j.second]
#         end
#     end
#     j = length(net.contractions)
#     for (i, oi) in enumerate(net.openidx)
#         # tensor leg must not participate in a contraction
#         @assert indexlist[oi.first][oi.second] == 0
#         # last qubit corresponds to fastest varying index
#         indexlist[oi.first][oi.second] = i - length(net.openidx) - 1 - j
#         leg_costs[abs(i - length(net.openidx) - 1 -j)] = size(net.tensors[oi.first])[oi.second]
#     end
#     # consistency check
#     for idx in indexlist, i in idx
#         @assert i != 0
#     end
#
#     return leg_costs, indexlist
# end


"""
    contract(net)
Fully contract a given TensorNetwork object.
"""

function contract(tn::TensorNetwork; optimize=false)
    indexlist, leg_costs, openidx = tn_to_ssa(tn)

    # TODO: approximate contraction using SVD splittings
    if optimize
        # TODO: add optimal search contraction (greedy / optimal)
        # sequence, cost = contract_order(net, legcosts, indexlist)
        # for i in 1:length(indexlist)
        #     for j in 1:length(indexlist[i])
        #         if indexlist[i][j] > 0
        #             indexlist[i][j] = findall(x->x==indexlist[i][j], sequence)[1]
        #         end
        #     end
        # end
        # return TensorOperations.ncon([t.data for t in net.tensors], indexlist; order=sequence)
    end

    # for now simply forward the contraction operation to 'ncon'
    return TensorOperations.ncon([t.data for t in tn.tensors], indexlist)
end


"""
    contract(tn::MPS, er::Real)

tn: TensorNetwork. Must be provided in a MPS form, that is, tn.tensors have three legs,
    tn.contractions are of the form[(T_i, 3), (T_i+1,1)]
er: maximum error in the truncation done in an individual contraction
"""
function contract(tn::MPS; er::Real=0.0)
    check_mps(tn)
    lchain = length(tn.tensors)
    tcontract = tn.tensors[1]
    k = length(size(tcontract)) - 2
    for j in 2:lchain
        tcontract = contract_svd(tcontract, tn.tensors[j], (j+k,1); er=er)
    end
    return tcontract.data
end
