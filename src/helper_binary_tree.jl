"""
    pred(t,w)

Output the predecessor nodes of node 't' in a tree with 'w' layers
"""
function pred(t,w)
    t > 1 || @error("This tensor has no predecessors in the tree.")

    k = layer(t,w)
    j = t-2^(k-1)+1
    if j%2 == 0
        tbef = 0.5(2^(k-1)-j)+j-1
        else tbef = 0.5(2^(k-1)-j-1)+j-1
    end
    return t - tbef-1
end

"""
    layer(j, w)
Return the layer to which node `j` belongs in a tree of `w` layers.
"""
function layer(j, w)
    if j ≤ 2^w-1
        k = ceil(log2(j+1))
    end
    return Int(k)
end

"""
    function_path(q1,q2,w)

Compute the path between qubits `q1` and `q2` in a binary tree tensor network of `w` layers. Here the path is a list of nodes that
join the nodes corresponding to `q1` and `q2`.
"""
function function_path(q1,q2,w)

    n = 2^(w-1) #number of qubits
    n_tensors = 2^w-1
    i = 2^w-1-q1+1
    j = 2^w-1-q2+1
    t1, t2 = copy(i), copy(j)
    pi = []
    pj = []

    pred_i, pred_j = 1,2
    while pred_i != pred_j
        pred_i = pred(i,w)
        push!(pi, pred_i)
        i = pred_i

        pred_j = pred(j,w)
        push!(pj, pred_j)
        j = pred_j
    end

    path = Int.([t1; pi; reverse(pj[1:end-1]); t2])
end
