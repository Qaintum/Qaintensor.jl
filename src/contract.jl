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
            j < length(net.contractions) || break
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
    for openidx in reverse(net.openidx)
        open_leg_number = -gen()
        tensors[openidx[1]][openidx[2]] = open_leg_number
        leg_costs[open_leg_number] = 1
    end
    return leg_costs, tensors
end


"""
    check_contraction!(Sab, Sa, Sb, leg_costs)

checks all combinations of tensors in Sa and Sb. Updates Sab for allowable combinations
"""

function check_contraction!(S, a, b, leg_costs, cr, Q, μs)
    μ_old = μs[1]
    μ_cap = μs[2]
    μ_next = μs[3]
    Sa = S[a]
    Sb = S[b]
    Sab = S[a+b]
    for Ta in Sa
        for Tb in Sb
            intersection = intersect(Ta, Tb)
            if length(intersection) == 0
                continue
            end

            if length(intersect(Q[(Ta,a)][1], Q[(Tb,b)][1])) != 0
                continue
            end

            Tnew = sort(unique(union(Ta, Tb)))
            μ = 1
            for leg in Tnew
                μ *= leg_costs[leg]
            end

            μ += Q[(Ta,a)][2]
            ordera = Q[(Ta,a)][1]

            μ += Q[(Tb,b)][2]
            orderb = Q[(Tb,b)][1]
            # if μ < Inf #μ_cap
            Qnew = union(ordera,orderb)

            if (Tnew, a+b) in keys(Q)
                if sort(Qnew) != sort(Q[(Tnew, a+b)][1])
                    push!(Sab, Tnew)
                end
                if μ < Q[(Tnew, a+b)][2]
                    Q[(Tnew, a+b)] = Qnew => μ
                end
            else
                Q[(Tnew, a+b)] = Qnew => μ
                push!(Sab, Tnew)
            end


            # else
            #     if μ_cap < μ < μ_next
            #         μ_next = μ
            #     end
            # end

        end
    end
    return μ_old, μ_cap, μ_next
end

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
    S = AbstractVector{AbstractVector}[]
    leg_costs, cr = contract_rep(net)

    Q = Dict{Tuple{AbstractVector, Int}, Pair{AbstractVector{Int}, Int}}()

    for (i,n) in enumerate(cr)
        # push!(L, ContractionCombination(n, 0, 0))
        Q[(n,1)] = [i] => 0
    end

    push!(S, copy(cr))
    for i in 2:n
        push!(S, AbstractVector[])
    end

    counter = 0
    μ_old = Inf
    μ_cap = Inf
    μ_next = Inf

    while length(S[n]) == 0
        for c in 2:n
            for d in 1:floor(Int, c//2)
                check_contraction!(S, d, c-d, leg_costs, cr, Q, (μ_old, μ_cap, μ_next))
                # println("Running")
                # println("Level : " * string(d) * ", " * string(c-d))
            end
        end
        # μ_old = μ_cap
        # μ_cap = μ_next
    end

    Tfinal = S[end][1]

    order = Q[(Tfinal, n)][1]
    # println(order)
    # order = [1, 6, 3, 2, 9, 4, 5, 7, 8, 10, 11]
    # order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # for x in keys(Q)
    #     if x[2] == n
    #         println(Q[x])
    #     end
    # end
    # println(net.contractions)

    # for i in 1:length(net.contractions)
    #     sum = net.contractions[i]
    #     loca = findall(x->x==sum.idx[1][1], order)[1]
    #     locb = findall(x->x==sum.idx[2][1], order)[1]
    #     net.contractions[i] = Summation([loca=>sum.idx[1][2], locb=>sum.idx[2][2]])
    # end
    #
    # for i in 1:length(net.openidx)
    #     openid = net.openidx[i]
    #     loca = findall(x->x==openid[1], order)[1]
    #     net.openidx[i] = loca=>openid[2]
    #     # openid[1] = order[openid[1]]
    # end
    # println(size.(net.tensors))
    # println(net.contractions)
    # print(Q)
    net.tensors = net.tensors[order]
    counter = 0
    # Create integer generator
    function gen()
        counter+=1
    end

    new_cr = deepcopy(cr)
    for i in 1:length(new_cr)
        for j in 1:length(new_cr[i])
            if new_cr[i][j] > 0
                new_cr[i][j] = 0
            end
        end
    end

    a = cr[order[1]]
    for i in 2:n
        b = cr[order[i]]
        for leg in intersect(a, b)
            index_counter = gen()
            for j in 1:i
                if leg in cr[order[j]]
                    index = findall(x->x==leg, cr[order[j]])[1]
                    new_cr[order[j]][index] = index_counter
                end
            end
        end
        a = unique(union(a,b))
    end

    return new_cr[order]
end

"""
    contract(net)

Fully contract a given TensorNetwork object.
"""

function contract(net::TensorNetwork; optimize=false)

    # TODO: approximate contraction using SVD splittings
    if optimize
        # println("A")
        copy_net = deepcopy(net)
        indexlist = contract_order(copy_net)
        # println(indexlist)
        @time a = TensorOperations.ncon([t.data for t in copy_net.tensors], indexlist)
        return a
        # return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
    end

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
    # println("B")
    # println(indexlist)
    @time b = TensorOperations.ncon([t.data for t in net.tensors], indexlist)
    # @time return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
    return b
end
