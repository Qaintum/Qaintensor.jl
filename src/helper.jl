"""
    cgc_to_ssa(cgc::CircuitGateChain{N})

obtain static single assignment form of circuit gate chain as list of connected indices
"""
function cgc_to_ssa(cgc::CircuitGateChain{N}; expectation_value=false) where {N}
    openidx = [1:N...]
    index_list = [deepcopy(openidx)]
    new_wire = N+1
    leg_costs = Dict()

    for moment in cgc
        for cg in moment
            gate_index = []
            for wire in sort([cg.iwire...])
                push!(gate_index, openidx[wire])
                push!(gate_index, new_wire)
                openidx[wire] = new_wire
                new_wire += 1
            end
            push!(index_list, sort(gate_index))
        end
    end
    push!(index_list, deepcopy(openidx))
    leg_costs = Dict{Int,Int}()
    for i in 1:new_wire-1
        leg_costs[i] = 2
    end
    index_list, leg_costs
end

"""
    tn_to_ssa(net::TensorNetwork)

obtain static single assignment form of tensor network as list of connected indices
"""
function tn_to_ssa(net::TensorNetwork)

    leg_costs = Dict{Int, Int}()
    openidx = Int[]
    indexlist = [fill(0, ndims(t)) for t in net.tensors]

    for (i, ts) in enumerate(net.contractions)
        for j in ts.idx
            @assert 1 ≤ j.second ≤ ndims(net.tensors[j.first])
            indexlist[j.first][j.second] = i
            leg_costs[i] = size(net.tensors[j.first])[j.second]
        end
    end
    j = length(net.contractions)
    for (i, oi) in enumerate(net.openidx)
        # tensor leg must not participate in a contraction
        @assert indexlist[oi.first][oi.second] == 0
        # last qubit corresponds to fastest varying index
        indexlist[oi.first][oi.second] = i - length(net.openidx) - 1 - j
        leg_costs[abs(i - length(net.openidx) - 1 -j)] = size(net.tensors[oi.first])[oi.second]
        push!(openidx, i - length(net.openidx) - 1 - j)
    end
    # consistency check
    for idx in indexlist, i in idx
        @assert i != 0
    end

    return indexlist, leg_costs, openidx
end

"""
    tn_graph_creation(index_list::Vector{Vector{T}}, leg_costs::Dict{Int, Int})

create LightGraph object representing tensor connections after decomposing given index_list into cannonical MPS form
"""
function tn_graph_creation(index_list::Vector, leg_costs::Dict{Int, Int}) where {T}
    g = SimpleGraph(length(index_list))
    edge_cost = Dict()
    for i in 1:length(index_list)
        for e in index_list[i]
            for j in i+1:length(index_list)
                if e in index_list[j]
                    add_edge!(g, i, j)
                    edge_cost[Set([i,j])] = leg_costs[e]
                end
            end
        end
    end
    g, edge_cost
end
