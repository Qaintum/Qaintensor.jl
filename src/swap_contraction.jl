# """
#     swap_index_decomposition(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int}) where {T}
#
# decomposes the given tensor network. assumes first and last tensors are input
# states, which will be decomposed into MPS form. all operators are split into
# MPO form
#
# """
# function swap_index_decomposition(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int}) where {T}
#     mindex_list = maximum([abs.(x) for x in index_list])
#     count = maximum(mindex_list)[1] + 1
#
#     new_index_list = []
#
#     t = popfirst!(index_list)
#
#     if length(t) > 2
#         num_t = length(t)
#         push!(new_index_list, [t[1], count])
#         leg_costs[count] = 2
#         for j in 2:num_t-1
#             push!(new_index_list, [t[j], count, count+1])
#             leg_costs[count+1] = leg_costs[count] * 2
#             count += 1
#         end
#         push!(new_index_list, [t[end], count])
#         count += 1
#     else
#         push!(new_index_list, t)
#     end
#
#     last_t = pop!(index_list)
#
#     while !isempty(index_list)
#         t = popfirst!(index_list)
#         if length(t) > 3
#             num_t = length(t)÷2
#             push!(new_index_list, [t[1], t[1+num_t], count])
#             leg_costs[count] = 2
#             for j in 2:num_t-1
#                 push!(new_index_list, [t[j], t[j+num_t], count, count+1])
#                 leg_costs[count+1] = leg_costs[count] * 2
#                 count += 1
#             end
#             push!(new_index_list, [t[num_t], t[end], count])
#             count += 1
#         else
#             push!(new_index_list, t)
#         end
#     end
#
#     t = last_t
#     if length(t) > 2
#         num_t = length(t)
#         push!(new_index_list, [t[1], count])
#         leg_costs[count] = 2
#         for j in 2:num_t-1
#             push!(new_index_list, [t[j], count, count+1])
#             leg_costs[count+1] = leg_costs[count] * 2
#             count += 1
#         end
#         push!(new_index_list, [t[end], count])
#         count += 1
#     else
#         push!(new_index_list, t)
#     end
#
#     new_index_list
# end

function contract_cost(open_idx, i, index1, index2, bond_dims, t_dims, max_bond_dim)
    abs(index1-index2) == 1 || error("Only supposed to contract legs when indices are adjacent in the MPS")
    cost = 0
    if index1 > index2
        hi = index1
        lo = index2
    else
        hi = index2
        lo = index1
    end

    if lo == 1
        cc_cost1 = 2*bond_dims[lo]
        cc_cost2 = 2*bond_dims[lo]*bond_dims[lo+1]
    elseif lo == length(open_idx) - 1
        cc_cost1 = 2*bond_dims[lo]*bond_dims[lo-1]
        cc_cost2 = 2*bond_dims[lo]
    else
        cc_cost1 = 2*bond_dims[lo]*bond_dims[lo-1]
        cc_cost2 = 2*bond_dims[lo]*bond_dims[lo+1]
    end

    if i == 1
        cc_cost1 *= t_dims[i]
        cc_cost2 *= t_dims[i]*t_dims[i+1]
    elseif i == length(t_dims)-1
        cc_cost1 *= t_dims[i]*t_dims[i-1]
        cc_cost2 *= t_dims[i]
    else
        cc_cost1 *= t_dims[i]*t_dims[i-1]
        cc_cost2 *= t_dims[i]*t_dims[i+1]
    end

    cost += cc_cost1 + cc_cost2

    bond_dims[lo] *= t_dims[i]

    if bond_dims[lo] > max_bond_dim
        if lo == 1
            cc_cost = 4*bond_dims[lo]*bond_dims[lo+1]
            svd_cost = 40*bond_dims[lo+1]
        elseif lo == length(open_idx) - 1
            cc_cost = 4*bond_dims[lo-1]*bond_dims[lo]
            svd_cost = 40*bond_dims[lo-1]
        else
            cc_cost = 4*bond_dims[lo-1]*bond_dims[lo]*bond_dims[lo+1]
            svd_cost = 40*min(bond_dims[lo-1]^2*bond_dims[lo+1], bond_dims[lo+1]^2*bond_dims[lo-1])
        end

        cost += cc_cost + svd_cost
        bond_dims[lo] = max_bond_dim
    end

    cost
end

function shift_adjacent!(open_idx, index1, index2, bond_dims, max_bond_dim)
    println("Shifting " * string(index2) * " to " * string(index1))
    δ = index2 - index1
    tmp = open_idx[index2:sign(-δ):index1 + sign(δ)]
    open_idx[index2:sign(-δ):index1 + sign(δ)] .= circshift(tmp, -1)
    # for index2:sign(-δ):index1
    #     tmp = open_idx[index1 + sign(δ)]
    #     open_idx[index1 + sign(δ)] = open_idx[index2]
    #     open_idx[index2] = tmp
    # end

    cost = 0
    mid = length(open_idx)÷2

    if index2 == length(open_idx)
        index2 -= sign(δ)
        cost += 8*bond_dims[index2] + 80*bond_dims[index2]
        if bond_dims[index2] > max_bond_dim
            bond_dims[index2] = max_bond_dim
        end
    end

    if index1 == length(open_idx)
        index1 += sign(δ)
        cost += 8*bond_dims[index1] + 80*bond_dims[index1]
        if bond_dims[index1] > max_bond_dim
            bond_dims[index1] = max_bond_dim
        end
    end

    for i in index2:sign(-δ):index1 + sign(δ)
        j = min(i, i - sign(δ))
        println("J is " * string(j))
        if j == 1
            c_cost = prod(bond_dims[j:j+1])
            svd_cost = bond_dims[j+1]
        elseif j >= length(open_idx) - 1
            c_cost = prod(bond_dims[j-1:j])
            svd_cost = bond_dims[j-1]
        else
            c_cost = prod(bond_dims[j-1:j+1])
            svd_cost = min(bond_dims[j-1]^2*bond_dims[j+1], bond_dims[j-1]*bond_dims[j+1]^2)
        end

        cost += 8*c_cost + 80*svd_cost
        if bond_dims[j] > max_bond_dim
            bond_dims[j] = max_bond_dim
        end
    end

    cost
end

function contract_legs!(open_idx, t, leg_cost, bond_dims, max_bond_dim)
    legs = intersect(open_idx, t)

    m = length(t)÷2
    t_dims =  vcat([2 .^ (m:-1:1)...], [2 .^ (2:m)]...)
    if isodd(length(t))
        t_dims = vcat(t_dims, [2^(m+1)...])
    end

    cost = 0
    index2 = 0
    n = length(t)÷2
    for i in 1:n-1
        println(open_idx)
        index1 = findfirst(x->x==t[i], open_idx)
        index2 = findfirst(x->x==t[i+1], open_idx)
        if abs(index1-index2) > 1
            cost += shift_adjacent!(open_idx, index1, index2, bond_dims, max_bond_dim)
        end
        index2 = index1 + sign(index2 - index1)

        cost += contract_cost(open_idx, i, index1, index2, bond_dims, t_dims, max_bond_dim)

        open_idx[index1] = t[i+n]
    end

    if index2 != 0
        open_idx[index2] = t[end]
    end

    cost
end

function swap_contraction_cost(cgc::CircuitGateChain{N}, max_bond_dim::Int) where {N}
    println(cgc)
    index_list, leg_costs = cgc_to_ssa(cgc)
    println(index_list)
    println(leg_costs)
    cost = 0
    open_idx = index_list[1]
    m = length(open_idx)÷2
    bond_dims = vcat([2 .^ (m:-1:1)...], [2 .^ (2:m)]...)
    if isodd(length(open_idx))
        bond_dims = vcat(bond_dims, [2^(m+1)...])
    end
    iter = 0
    for t in index_list[2:end]
        # println(t)
        println(open_idx)
        cost += contract_legs!(open_idx, t, leg_cost, bond_dims, max_bond_dim)
    end

    for i in length(bond_dims)-1
        cost += bond_dims[i] * bond_dims[i+1]
    end
    cost += bond_dims[end]
    cost
end
