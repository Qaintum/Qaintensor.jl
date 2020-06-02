"""
    contraction_rep(net::TensorNetwork)

Switch tensors to get S representation for contraction optimization
"""
function contract_rep(net::TensorNetwork)
    counter = 0
    # Create integer generator
    function gen()
        counter+=1
    end

    tensors = AbstractArray[]
    leg_costs = Dict{Int, Int}()
    j = 1
    for i in 1:length(net.tensors)
        einsum_rep = Vector(undef, ndims(net.tensors[i]))
        while net.contractions[j].idx[2][1] == i
            einsum_rep[net.contractions[j].idx[2][2]] = tensors[net.contractions[j].idx[1][1]][net.contractions[j].idx[1][2]]
            j <= length(net.tensors) || break
            j += 1
        end
        tensor_dims = size(net.tensors[i])
        for k in 1:length(einsum_rep)
            if !isassigned(einsum_rep, k)
                leg_number = gen()
                einsum_rep[k] = leg_number
                leg_costs[leg_number] = tensor_dims[k]
            end
        end
        push!(tensors, einsum_rep)
    end
    counter = 0
    for openidx in net.openidx
        tensors[openidx[1]][openidx[2]] = -gen()
    end
    return leg_costs, tensors
end

# """
#     contract_cost(tensors::AbstractVector{Tensor})
#
# function to calculate cost of performing a single contraction
# """
# function contract_cost(tensors::AbstractVector{Tensor})
#
# end

"""
    ContractionCombination

Datastructure to represent a possible contraction combination
"""
struct ContractionCombination
    I::AbstractVector{Int}
    ζ::Int
    f::Int
end

"""
    contract_order(net::TensorNetwork)

simple port of contract_path algorithms from github.com/dgasmith/opt_einsum to julia
"""

function contract_order(net::TensorNetwork)
    n = length(net.tensors)
    S = AbstractVector[]
    L = ContractionCombination[]
    leg_costs, cr = contract_rep(net)
    Q = Dict{AbstractVector{Int}, Pair{AbstractVector{Int}, Int}}

    for (n,T) in zip(cr, net.tensors)
        push!(L, ContractionCombination(n, 0, 0))
    end

    push!(S, cr)
    for i in 2:n
        push!(S, Pair[])
    end
    μ_old = 0
    μ_cap = 1
    μ_next = Inf

    for c in 2:n
        for d in 1:floor(Int, c//2)
            # check_combination!(S[2c-d], S[c], S[c-d], leg_costs)
        end
        if length(S[n]) != 0
            break
        end
    end
    return Q
end

"""
    contract(net)

Fully contract a given TensorNetwork object.
"""

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
