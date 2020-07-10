"""
    default_cost_function(g, e, edge_costs)

greedy cost function that minimizes largest bond dimension first.
"""
function default_cost_function(g, e, edge_costs)
    cost = edge_costs[Set([src(e), dst(e)])]
    contraction_cost = 1
    for v in neighbors(g, src(e))
        contraction_cost *= edge_cost[Set([src(e), v])]
    end
    for v in neighbors(g, dst(e))
        contraction_cost *= edge_cost[Set([dst(e), v])]
    end
    contraction_cost /= edge_cost[Set([src(e), dst(e)])]
    cost, contraction_cost
end

"""
    min_edge_search(g, edge_cost, color; cost_func)

searches for and returns edge contraction with lowest overall cost, cost of contraction and mininized output of cost function
"""

function min_edge_search(g, edge_costs, color; cost_func=default_cost_function)
    min_cost = typemax(Int64)
    min_edge = nothing
    min_contraction_cost = nothing
    internal = false
    for e in edges(g)
        cost, contraction_cost = cost_func(g, e, edge_costs)
        if cost < min_cost && color[src(e)] != color[dst(e)]
            min_edge = Edge(src(e), dst(e))
            min_cost = cost
            min_contraction_cost = contraction_cost
        end
    end
    min_edge, min_cost, min_contraction_cost
end

"""
    cost_estimation(T::TensorNetwork; type::String="mps_contract")

estimates the cost of contraction heuristics
"""
function cost_estimate(cgc::CircuitGateChain{N}) where {N}
    size_dict, rep, openidx = contract_rep(T)
    println(size_dict)
    println(rep)
    println(openidx)
end
