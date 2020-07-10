"""
    swap_index_decomposition(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int}) where {T}

decomposes the given tensor network. assumes first and last tensors are input
states, which will be decomposed into MPS form. all operators are split into
MPO form

"""
function swap_index_decomposition(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int}) where {T}
    mindex_list = maximum([abs.(x) for x in index_list])
    count = maximum(mindex_list)[1] + 1

    new_index_list = []

    t = popfirst!(index_list)

    if length(t) > 2
        num_t = length(t)
        push!(new_index_list, [t[1], count])
        leg_costs[count] = 2
        for j in 2:num_t-1
            push!(new_index_list, [t[j], count, count+1])
            leg_costs[count+1] = leg_costs[count] * 2
            count += 1
        end
        push!(new_index_list, [t[end], count])
        count += 1
    else
        push!(new_index_list, t)
    end

    last_t = pop!(index_list)

    while !isempty(index_list)
        t = popfirst!(index_list)
        if length(t) > 3
            num_t = length(t)÷2
            push!(new_index_list, [t[1], t[1+num_t], count])
            leg_costs[count] = 2
            for j in 2:num_t-1
                push!(new_index_list, [t[j], t[j+num_t], count, count+1])
                leg_costs[count+1] = leg_costs[count] * 2
                count += 1
            end
            push!(new_index_list, [t[num_t], t[end], count])
            count += 1
        else
            push!(new_index_list, t)
        end
    end

    t = last_t
    if length(t) > 2
        num_t = length(t)
        push!(new_index_list, [t[1], count])
        leg_costs[count] = 2
        for j in 2:num_t-1
            push!(new_index_list, [t[j], count, count+1])
            leg_costs[count+1] = leg_costs[count] * 2
            count += 1
        end
        push!(new_index_list, [t[end], count])
        count += 1
    else
        push!(new_index_list, t)
    end

    new_index_list
end

function shift_adj!(open_idx, index1, index2)
    tmp = open_idx[index1 - sign(index1-index2)]
    open_idx[index1 - sign(index1-index2)] = open_idx[index2]
    open_idx[index1] = tmp
    cost = 0
    mid = length(open_idx)÷2
    for i in index2:sign(index1-index2):index1
        n = abs(mid-i)
        cost += 2^(3n+2) + 10*2^(3n+1)
    end
    cost
end

function contract_legs!(open_idx, t, legs)
    legs = intersect(open_idx, t)
    cost = 0
    for i in 1:length(legs)-1
        index1 = findfirst(x->x==legs[i], open_idx)
        index2 = findfirst(x->x==legs[i+1], open_idx)
        if abs(index1-index2) > 1
            cost += shift_adjacent!(open_idx, index1, index2)
        end
        index2 = findfirst(x->x==legs[i+1], open_idx)

        cost += contract_cost(open_idx, index1, index2)
    end
    cost
end

function swap_contraction_cost(cgc::CircuitGateChain{N}, max_bond_dim::Int) where {N}
    index_list, leg_costs = cgc_to_ssa(cgc)
    println(index_list)
    println(leg_costs)
    cost = 0
    open_idx = index_list[1]
    for t in index_list[2:end]

        cost += contract_leg!(open_idx, t, legs, leg_cost)
    end
    # index_list, leg_costs = index_creation(cgc)
    # index_list = swap_index_decomposition(index_list, leg_costs)
    # g, edge_costs = tn_graph_creation(index_list, leg_costs)
    # print(edge_costs)
    iter = 0
    t = TikzGraphs.plot(g)
    TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)
    cost = 0
end
