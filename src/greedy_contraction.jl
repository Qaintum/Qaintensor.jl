using DataStructures

function memory_removed(size12::Int, size1::Int, size2::Int, k12, k1, k2)
    size12 - size1 - size2
end

COST_FUNCTIONS = Dict([("memory-removed", memory_removed)])

"""
    contraction_chooser()

Function that chooses a possible contraction from a given selection of trial contractions. These are distributed
using a Boltzmann's distribution
"""
function contraction_chooser()
end


function _simple_chooser(queue::PriorityQueue, remaining)
    """Default contraction chooser that simply takes the minimum cost option.
    """
    cost, k1, k2, k12 = dequeue!(queue)
    if !(k1 in remaining) || !(k2 in remaining)
        return None  # candidate is obsolete
    end
    cost, k1, k2, k12
end

"""
    greedy_optimize(input, output, size_dict; choose_fn=None, cost_fn="memory-removed")

Function that greedily finds contraction path for opt_einsum.
from Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753
"""

function greedy_contraction_order(inputs, outputs, size_dict, choose_fn; cost_fn="memory-removed")
    if length(input) == 1
        return [(0,)]
    end

    cost_fn = COST_FUNCTIONS[cost_fn]

    # set the function that chooses which contraction to take
    if isnothing(choose_fn)
        choose_fn = _simple_chooser
        push_all = false
    else
        # assume chooser wants access to all possible contractions
        push_all = true
    end

    remaining = Dict()
    ssa_ids = Iterators.Stateful(1:length(inputs))

    ssa_path = Tuple{Int, Int}[]

    for (ssa_id, key) in enumerate(inputs)
        if key in remaining
            push!(ssa_path, (remaining[key], ssa_id))
            remaining[key] = popfirst!(ssa_ids)
        else
            remaining[key] = ssa_id
        end
    end

end

"""
    random_greedy_optimize(r, inputs, outputs, size_dict, choose_fn, cost_fn)

implementation of the greedy optimize algorithm for opt_einsum.
from Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753
"""
function random_greedy_optimize(r, inputs, outputs, size_dict, choose_fn, cost_fn)
    if r == 0
        choose_fn = nothing
    end
end
