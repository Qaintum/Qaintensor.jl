using LightGraphs, SimpleWeightedGraphs
using TikzGraphs
using Statistics

"""
    catn_index_decomposition(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int}) where {T}

decomposes the given tensor network into cannonical MPS form, such that each tensor is only composed of MPSs
"""
function catn_index_decomposition(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int}; max_bond_dim=typemax(Int64)) where {T}
    mindex_list = maximum([abs.(x) for x in index_list])
    count = maximum(mindex_list)[1] + 1

    new_index_list = []
    color = []
    num = 1
    while !isempty(index_list)
        t = popfirst!(index_list)
        if length(t) > 2
            num_t = length(t)
            push!(new_index_list, [t[1], count])
            push!(color, num)
            leg_costs[count] = 2
            for j in 2:num_t-1
                push!(new_index_list, [t[j], count, count+1])
                push!(color, num)
                leg_costs[count+1] = leg_costs[count] * 2 < max_bond_dim ? leg_costs[count] * 2 : max_bond_dim
                count += 1
            end
            push!(new_index_list, [t[end], count])
            push!(color, num)
            count += 1
        else
            push!(new_index_list, t)
            push!(color, num)
        end
        num += 1
    end
    new_index_list, color
end


"""
    mps_cost_function(g, e, edge_costs)

returns cost of contracting along chosen edge. implemented from Pan et. al, arXiv:1912.03014v1
"""
function mps_cost_function(g, e, edge_costs)
    cost = 0
    contraction_cost = 1
    for v in neighbors(g, src(e))
        cost += log(edge_costs[Set([src(e), v])])
        contraction_cost *= edge_costs[Set([src(e), v])]
    end
    for v in neighbors(g, dst(e))
        cost += log(edge_costs[Set([dst(e), v])])
        contraction_cost *= edge_costs[Set([dst(e), v])]
    end
    cost -= 2*log(edge_costs[Set([src(e), dst(e)])])
    contraction_cost /= edge_costs[Set([src(e), dst(e)])]
    cost, contraction_cost
end

"""
    swap_cost(g, v1, v2, edge_costs)

calculates the cost of swapping legs positions in an MPS in canonical form.
performs a contraction along the shared edge, a permutation of indices, followed
 by an SVD
"""
function swap_cost(g, v1, v2, edge_costs)
    v1 in neighbors(g, v2) || error("Only able to swap neighboring vertices")
    ccost1 = 1
    for v in neighbors(g, v1)
        ccost1 *= edge_costs[Set([v1, v])]
    end
    ccost2 = 1
    for v in neighbors(g, v2)
        ccost2 *= edge_costs[Set([v2, v])]
    end

    contraction_cost = ccost1*ccost2/edge_costs[Set([v1, v2])]
    svd_cost = 10*minimum([ccost1*ccost2*ccost2, ccost1*ccost1*ccost2])/edge_costs[Set([v1, v2])]/edge_costs[Set([v1, v2])]
    contraction_cost + svd_cost
end

"""
    swap_edges(g, v1, v2, edge_costs)

swaps the edges
"""
function swap_edges(g, mps, target, mps_color, color, edge_costs)
    edges_to_add =  []

    for nb in neighbors(g, mps)
        if color[nb] != mps_color
            if has_edge(g, Edge(mps, nb))
                rem_edge!(g, Edge(mps, nb))
                ea = Edge(target, nb)
            else
                rem_edge!(g, Edge(nb, mps))
                ea = Edge(nb, target)
            end
            println("Removing " * string(Set([mps, nb])))
            println("Adding " * string(Set([target, nb])))
            edge_costs[Set([target, nb])] = edge_costs[Set([mps, nb])]
            delete!(edge_costs, Set([mps, nb]))
            push!(edges_to_add, ea)
        end
    end

    for nb in neighbors(g, target)
        if color[nb] != mps_color
            if has_edge(g, Edge(target, nb))
                rem_edge!(g, Edge(target, nb))
                ea = Edge(mps, nb)
            else
                rem_edge!(g, Edge(nb, target))
                ea = Edge(nb, mps)
            end
            println("Removing " * string(Set([target, nb])))
            println("Adding " * string(Set([mps, nb])))
            edge_costs[Set([mps, nb])] = edge_costs[Set([target, nb])]
            delete!(edge_costs, Set([target, nb]))
            push!(edges_to_add, ea)
        end
    end

    for edge in edges_to_add
        add_edge!(g, edge)
    end
end

"""
    shift_to_edge!(g, edge_costs, mps, color, final_vertex; head=true)

shifts given mps to edge of tensor before contracting two different tensors
"""
function shift_to_edge!(g, edge_costs, mps, color, final_vertex)

    if length(neighbors(g, mps)) <= 2
        push!(final_vertex, mps)
        return 0
    end
    mps_color = color[mps]
    targets = []

    for i in 1:length(color)
        if color[i] == mps_color && length(neighbors(g, i)) == 2
            push!(targets, i)
        end
        if length(targets) == 2
            break
        end
    end

    function h_func(n)
        start_color = color[mps]
        if color[n] != color[mps]
            return typemax(Int16)
        end
        return 1
    end

    swapcost = 0

    path1 = a_star(g, mps, targets[1], LightGraphs.DefaultDistance(), h_func)
    path2 = a_star(g, mps, targets[2], LightGraphs.DefaultDistance(), h_func)

    if length(path1) < length(path2)
        path = path1
    else
        path = path2
    end

    if length(path) < 1
        push!(final_vertex, mps)
        return swapcost
    end

    # if head
    #     i = -1
    # else
    #     i = 1
    # end
    # swapcost = 0
    #
    # target = mps + i
    # println("Target " * string(target) * " MPS: " * string(mps))
    # println(color)

    for e in path
        src_vertex = src(e)
        dst_vertex = dst(e)
        swapcost += swap_cost(g, src_vertex, dst_vertex, edge_costs)
        swap_edges(g, src_vertex, dst_vertex, mps_color, color, edge_costs)
    end
    push!(final_vertex, dst(path[end]))
    println("Shifting " * string(mps) * " to " * string(dst(path[end])) * "\nCost increased by " * string(swapcost))
    swapcost

    #
    # while length(neighbors(g, mps)) > 2
    #     println("Now swapping " * string(mps) * " and " * string(mps+i))
    #
    #     if isempty(neighbors(g, target)) || color[target] != mps_color || !(target in neighbors(g, mps))
    #         i += sign(i)
    #         target = mps + i % length(color)
    #         if target <= 0
    #             target += length(color)
    #         end
    #         continue
    #     end
    #     target = mps + i % length(color)
    #     if target <= 0
    #         target += length(color)
    #     end
    #     swapcost += swap_cost(g, mps, target, edge_costs)
    #     swap_edges(g, mps, target, mps_color, color, edge_costs)
    #     mps = target
    #     i = Int(i/norm(i))
    # end
    # push!(final_vertex, mps)
    # swapcost
end

"""
    shift_adjacent!(g, edge_costs, mps, color, final_vertex)

shifts given mps adjacent to target mps
"""
function shift_adjacent!(g, edge_costs, mps, target, color, final_vertex)
    function h_func(n)
        start_color = color[mps]
        if color[n] != color[mps]
            return typemax(Int16)
        end
        return 1
    end
    swapcost = 0
    path = a_star(g, mps, target, LightGraphs.DefaultDistance(), h_func)
    println(path)
    if length(path) <= 1
        push!(final_vertex, mps)
        return swapcost
    end
    mps_color = color[mps]
    for e in path[1:end-1]
        src_vertex = src(e)
        dst_vertex = dst(e)
        swapcost += swap_cost(g, src_vertex, dst_vertex, edge_costs)
        swap_edges(g, src_vertex, dst_vertex, mps_color, color, edge_costs)
    end
    push!(final_vertex, src(path[end]))
    println("Shifting " * string(mps) * " to " * string(target) * "\nCost increased by " * string(swapcost))
    swapcost
end

function contract_cost(g, v1, v2, edge_costs)
    ccost1 = 1
    for v in neighbors(g, v1)
        println(v)
        ccost1 *= edge_costs[Set([v1, v])]
        println("Ccost1 " * string(ccost1))
    end
    ccost2 = 1
    for v in neighbors(g, v2)
        println(v)
        ccost2 *= edge_costs[Set([v2, v])]
        println("Ccost2 " * string(ccost2))
    end
    contraction_cost = ccost1*ccost2/edge_costs[Set([v1, v2])]
end

function contract_mps!(g, srcv, dstv, edge_costs)
    if srcv == dstv
        return 0
    end
    cost = contract_cost(g, srcv, dstv, edge_costs)
    println("Contracting edge " * string(Set([srcv, dstv])))
    rem_edge!(g, Edge(srcv, dstv))
    delete!(edge_costs, Set([srcv, dstv]))
    for nb in neighbors(g, dstv)
        if has_edge(g, Edge(dstv, nb))
            rem_edge!(g, Edge(dstv, nb))
            add_edge!(g, Edge(srcv, nb))
        else
            rem_edge!(g, Edge(nb, dstv))
            add_edge!(g, Edge(nb, srcv))
        end
        println("Edge costs of " * string(Set([dstv, nb])) * " shifted to " * string(Set([srcv, nb])))
        edge_costs[Set([srcv, nb])] = edge_costs[Set([dstv, nb])]
        delete!(edge_costs, Set([dstv, nb]))
        # println("Removing " * string(Set([dstv, nb])))
    end
    println("Contracting " * string(srcv) * " with " * string(dstv) * "\nCost increased by " * string(cost))
    cost
end

function check_legs(g, color, mps_color)
    legs = []
    connected_colors = []
    matching_color = nothing
    first_color = Dict()

    for i in 1:length(color)
        if color[i] == mps_color
            for nb in neighbors(g, i)
                if isnothing(matching_color)
                    if color[nb] != mps_color && color[nb] in connected_colors
                        matching_color = color[nb]
                        push!(legs, first_color[matching_color])
                        push!(legs, i)
                    elseif color[nb] != mps_color
                        push!(connected_colors, color[nb])
                        first_color[color[nb]] = i
                    end
                    continue
                end

                if color[nb] == matching_color
                    push!(legs, i)
                end
            end
        end
    end
    legs, matching_color
end

function merge_leg(g, leg1, leg2, leg3, leg4, edge_costs)
    if leg1 == leg2 && leg3 == leg4
        return 0
    end

    cost = 0
    if has_edge(g, Edge(leg2, leg4))
        rem_edge!(g, Edge(leg2, leg4))
    else
        rem_edge!(g, Edge(leg4, leg2))
    end
    println("Multiplying " * string(Set([leg1, leg3])) * " with " * string(Set([leg2, leg4])))
    edge_costs[Set([leg1, leg3])] *= edge_costs[Set([leg2, leg4])]
    println("Edge cost for " * string(Set([leg1, leg3])) * " increased to " * string(edge_costs[Set([leg1, leg3])]))
    delete!(edge_costs, Set([leg2, leg4]))

    cost += contract_mps!(g, leg1, leg2, edge_costs)
    cost += contract_mps!(g, leg3, leg4, edge_costs)

    cost
end

function merge_legs(g, legs, color, matching_color, edge_costs, max_bond_dim)
    med_leg = sort(legs)[Int(round(length(legs)/2))]
    leg3 = filter(x->color[x]==matching_color, neighbors(g, med_leg))[1]
    cost = 0
    # println("Merge into " * string(med_leg) * " and " * string(leg3))
    # Move legs close to median leg (try to minimize shifts)

    for leg in legs
        leg4 = filter(x->color[x]==matching_color, neighbors(g, leg))[end]
        println("Merging " * string(leg) * " and " * string(leg4) * " into " * string(med_leg) * " and " * string(leg3))
        med_target = []
        leg_target = []
        cost += shift_adjacent!(g, edge_costs, leg4, leg3, color, leg_target)
        cost += shift_adjacent!(g, edge_costs, leg, med_leg, color, med_target)

        # println("Merged legs are " * string(med_leg) * " & " * string(med_leg-sign(med_leg - leg)*med_i) * ", " * string(leg3) * " & " * string(leg3-sign(leg3 - leg4)*leg_i))
        cost += merge_leg(g, med_leg, med_target[1], leg3, leg_target[1], edge_costs)

    end

    # Perform SVD to reduce dimension
    if edge_costs[Set([med_leg, leg3])] > max_bond_dim
        println("Performing SVD on edge " * string(Set([med_leg, leg3])))
        ccost1 = 1
        for v in neighbors(g, med_leg)
            ccost1 *= edge_costs[Set([med_leg, v])]
        end
        ccost2 = 1
        for v in neighbors(g, leg3)
            ccost2 *= edge_costs[Set([leg3, v])]
        end
        contraction_cost = ccost1*ccost2/edge_costs[Set([leg3, med_leg])]
        cost += 10*minimum([ccost1*ccost2*ccost2, ccost1*ccost1*ccost2])/edge_costs[Set([leg3, med_leg])]/edge_costs[Set([leg3, med_leg])]
        edge_costs[Set([med_leg, leg3])] = max_bond_dim
    end
    cost
end


function catn_contraction_cost(cgc::CircuitGateChain{N}, max_bond_dim::Int) where {N}
    max_bond_dim = 6
    # index_list, leg_costs = index_creation(cgc)
    index_list, leg_costs = cgc_to_ssa(cgc)
    index_list, color = catn_index_decomposition(index_list, leg_costs; max_bond_dim=max_bond_dim)
    g, edge_costs = tn_graph_creation(index_list, leg_costs)

    iter = 0
    t = TikzGraphs.plot(g)
    TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)
    cost = 0
    # Merge bonds together before performing contractions
    for tensor_color in unique(color)
        legs, matching_color = check_legs(g, color, tensor_color)
        println("Legs to merge include " * string(legs))

        while !isnothing(matching_color)
            cost += merge_legs(g, legs, color, matching_color, edge_costs, max_bond_dim)
            legs, matching_color = check_legs(g, color, tensor_color)
            iter += 1
            t = TikzGraphs.plot(g)
            println(edge_costs)
            TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)
        end

    end

    iter += 1

    t = TikzGraphs.plot(g)
    println(edge_costs)
    TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)

    # Find good edge to contract
    # for i in 1:3
    while length(unique(color)) > 1
        min_edge, min_cost, contraction_cost = min_edge_search(g, edge_costs, color; cost_func=mps_cost_function)
        # min_edge, min_cost, contraction_cost = cost_function(g, edge_costs, color)
        mps1 = src(min_edge)
        mps2 = dst(min_edge)

        println("Contracting min leg " * string(min_edge))
        # Shift nodes to merge to edge of MPS
        src_vertex = []
        dst_vertex = []
        cost += shift_to_edge!(g, edge_costs, mps1, color, src_vertex)
        cost += shift_to_edge!(g, edge_costs, mps2, color, dst_vertex)

        # Contract edge
        cost += contract_mps!(g, src_vertex[1], dst_vertex[1], edge_costs)

        t = TikzGraphs.plot(g)
        iter += 1
        println(edge_costs)
        TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)

        old_color = color[mps2]
        # Two MPS become one MPS

        color_indices =  findall(x->x==old_color, color)

        color[color_indices] .= color[mps1]

        for key in keys(edge_costs)
            if color[mps2] in key
                new_key = deepcopy(key)
                delete!(new_key, old_color)
                push!(new_key, color[mps1])
            end
        end

        # Merge bonds together
        legs, matching_color = check_legs(g, color, color[mps1])
        println("Legs to merge include " * string(legs))
        merged = false
        while !isnothing(matching_color)
            merged = true
            cost += merge_legs(g, legs, color, matching_color, edge_costs, max_bond_dim)
            legs, matching_color = check_legs(g, color, color[mps1])
        end
        if merged
            iter += 1
            t = TikzGraphs.plot(g)
            TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)
        end
    end
    # contract remaining MPS
    println("Contracting remaining MPS")
    for i in 1:length(color)
        nb = neighbors(g, i)
        if !isempty(nb)
            ne = nb[1]
            cost += contract_cost(g, i, ne, edge_costs)
            if has_edge(g, Edge(i, ne))
                rem_edge!(g, Edge(i, ne))
            else
                rem_edge!(g, Edge(ne, i))
            end
            println(nb)
            delete!(edge_costs, Set([ne, i]))
        end
    end
    t = TikzGraphs.plot(g)
    println(edge_costs)
    iter += 1
    TikzGraphs.save(TikzGraphs.SVG("simplegraph" * string(iter) * ".svg"), t)
    cost
end
