using LightGraphs
using StatsBase: sample

function ⊂(A,B)
    for a in A
        (a ∉ B) ? (return false) : nothing
    end
    return true
end

"""
    random_graph(Nn, Ne)

Random graph with Nn nodes and at most Ne edges (multiedges are not supported)
"""
function random_graph(Nn, Ne)
    G = Graph(Nn)
    possible_edges = Tuple{Int, Int}[]
    for i in 1:Nn, j in 1:i-1
        push!(possible_edges, (i,j))
    end
    edges = sample(possible_edges, Ne, replace = false)
    for e in edges
        add_edge!(G, e[1], e[2])
    end
    return G
end

# Interaction graph of circuit with only k-neighbors interactions
function local_circuit_graph(N, k)
    G = Graph(N)
    for i in 1:N-1
        for j in 1:min(k-1, N-i)
            LightGraphs.add_edge!(G, i, i+j)
        end
    end
    return G
end


"""
    network_graph(net)

Graph of TensorNetwork  `net` using JuliaGraphs. Returns the graph and
a dictionary `edge_idx` that for a given pair of Tensors `(i, j)` yields
the indices of `net.contractions` that involve them.
"""
function network_graph(net::TensorNetwork)
    M = length(net.tensors)
    G = Graph(M)
    edge_idx = Dict{Tuple{Int,Int}, Array{Int,1}}()

    for (k, s) in enumerate(net.contractions)
        length(s.idx) == 2 || error("Contractions of more than 2 tensors not supported")
        i::Int, j::Int = s.idx[1].first, s.idx[2].first
        i ≤ j ? nothing : (i, j) = (j, i)
        LightGraphs.add_edge!(G, i, j)

        # add index for later contraction
        if (i, j) in keys(edge_idx)
            push!(edge_idx[(i,j)], k)
        else
            edge_idx[(i,j)] = [k]
        end
    end

    return G, edge_idx
end

"""
    line_graph(G)

Line graph of G.
"""
function line_graph(G::Graph)
    LG = Graph()
    nodeinfo = []

    # add nodes
    for i in 1:nv(G)
        neigh = G.fadjlist[i]
        for j in neigh
            # add the node if not present
            edge = (i < j)  ? (i, j) : (j, i)
            if edge ∉ nodeinfo
                add_vertex!(LG)
                push!(nodeinfo, edge)
            end
        end
    end

    # add edges
    for (i, e1) in enumerate(nodeinfo)
        for (j, e2) in enumerate(nodeinfo[i+1:end])
            common = e1 ∩ e2
            if length(common) > 0
                LightGraphs.add_edge!(LG, i, j + i)
            end
        end
    end
    return LG, nodeinfo
end

"""
    line_graph(net)

Line graph of a Tensor Network. Multiple edges between same nodes
are represented as different nodes of the line_graph
# Returns
- `LG::Graph`: line graph without metainformation
- `nodeinfo::Array{NTuple{3, Int64}}`: `nodeinfo[i] = (n1, n2, idx)` contains
info about the contracted leg of `net` that is represented as node `i` of `LG`.
Thus, `n1` and `n2` are the indices of the Tensors connected by `Summation[idx]`.
All open indices are disregarded.
"""
function line_graph(net::TensorNetwork)
    (length(net.openidx) == 0) || @warn("All open indices are disregarded")
    G, edge_idx = network_graph(net)

    LG = Graph()
    nodeinfo = NTuple{3, Int64}[]

    # add nodes
    for i in 1:nv(G)
        neigh = G.fadjlist[i]
        for j in neigh[neigh .> i]
            # get number of parallel edges
            for e in edge_idx[(i, j)]
                add_vertex!(LG)
                push!(nodeinfo, (i, j, e))
            end
        end
    end

    # add edges
    for (i, e1) in enumerate(nodeinfo)
        for (j, e2) in enumerate(nodeinfo[i+1:end])
            common = (e1[1], e1[2]) ∩ (e2[1], e2[2])
            if length(common) > 0
                LightGraphs.add_edge!(LG, i, j + i)
            end
        end
    end
    return LG, nodeinfo
end


"""
    interaction_graph(cgc)

Interaction graph of a CircuitGateChain.
# Returns
- `G::Graph`: a graph with `N` nodes. Nodes `i` and `j` are connected iff
there is a gate in `cgc` acting on both.
"""
function interaction_graph(cgc::CircuitGateChain{N}) where N
    G = Graph(N)
    for moment in cgc
        for cg in moment
            for (j, i1) in enumerate(cg.iwire)
                for i2 in cg.iwire[1:j-1]
                    LightGraphs.add_edge!(G, i1, i2)
                end
            end
        end
    end
    return G
end

"""
    lacking_for_clique_neigh(G::Graph, i)

Finds the edges that are lacking to `G` for the neighborhood of vertex `i`
to be a clique. Returns this number of edges and the edges themselves.
"""
function lacking_for_clique_neigh(G::Graph, i::Int)
    neigh = G.fadjlist[i]
    lacking = Tuple{Int,Int}[]
    for (j, i2) in enumerate(neigh)
        for i1 in neigh[1:j-1]
            has_edge(G, i1, i2) ? nothing : push!(lacking, (i1, i2))
        end
    end
    return length(lacking), lacking
end

"""
    rem_vertex_fill!(G, i, lacking, ordering, vertex_label)

Fills `G` with the lacking edges for the neighborhood of `i` to be a clique,
removes vertex `i` of `G`, pushes it to `ordering` and updates `vertex_label`.
LightGraphs changes the order of the vertices when one is removed like this:

removed                  moved
   |                       |
  [v1    v2    v3    v4    v5]    -->    [v5    v2    v3    v4]
"""
function rem_vertex_fill!(G::Graph, i::Int, lacking::Vector{Tuple{Int,Int}},
                          ordering::Vector{Int}, vertex_label)
    for e in lacking
        LightGraphs.add_edge!(G, e[1], e[2])
    end
    push!(ordering, vertex_label[i])
    rem_vertex!(G, i)
    v = pop!(vertex_label)
    if i ≤ nv(G)
        vertex_label[i] = v
    end
end


"""
    min_fill_ordering(G)

Find an ordering of the vertices of `G` using the min-fill heuristic
(cfr. [Bodlaender, _Discovering Treewidth_](http://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2005-018.pdf))
"""
function min_fill_ordering(G::Graph)
    H = copy(G)
    ordering = Int[]
    no_lacking = Tuple{Int,Int}[] # to pass when there is no edge lacking
    vertex_label = collect(1:nv(H))
    while nv(H) > 0
        success = false
        for i in nv(H):-1:1
            if length(H.fadjlist[i]) == 0
                rem_vertex_fill!(H, i, no_lacking, ordering, vertex_label)
                success = true
            end
        end
        for i in nv(H):-1:1
            if length(H.fadjlist[i]) == 1
                rem_vertex_fill!(H, i, no_lacking, ordering, vertex_label)
                success = true
            end
        end

        if ! success
            degrees = length.(H.fadjlist)
            J = sortperm(degrees)

            found_clique = false
            v = 0
            best_n_lacking = Inf
            best_lacking = Tuple{Int,Int}[]
            for i in 1:nv(H)
                j = J[i]
                n_lacking, lacking = lacking_for_clique_neigh(H, j)
                if n_lacking == 0
                    rem_vertex_fill!(H, j, lacking, ordering, vertex_label)
                    found_clique = true
                    break
                elseif n_lacking < best_n_lacking
                    v = j
                    best_n_lacking = n_lacking
                    best_lacking = lacking
                end
            end
            # if running until here, remove v
            if ! found_clique
                rem_vertex_fill!(H, v, best_lacking, ordering, vertex_label)
            end
        end
    end
    ordering
end

"""
    triangulation(G::Graph, ordering)

Chordal completion of `G` following order `ordering`.
For each vertex add edges between its higher numbered neighbors.
"""
function triangulation(G::Graph, ordering)
    H = copy(G)
    for (i, v) in enumerate(ordering)
        neigh = H.fadjlist[v]
        high_neigh = neigh ∩ ordering[i+1:end] # get higher numbered neighbors
        for (j, i1) in enumerate(high_neigh)
            for i2 in high_neigh[1:j-1]
                LightGraphs.add_edge!(H, i1, i2)
            end
        end
    end
    return H
end

"""
    tree_decomposition(G)

Finds a tree decomposition of G using the min-fill heuristic. Returns the
width of the decomposition, the tree and the bags.
"""
function tree_decomposition(G::Graph)
    ordering = min_fill_ordering(G)
    H = triangulation(G, ordering)
    up_neighs = [H.fadjlist[ordering[i]] ∩ ordering[i+1:end] for i in 1:nv(H)]
    # first bag is the first node within a maximum clique following the ordering
    clique_neigh_idx = findfirst(length.(up_neighs) .== nv(H)-1:-1:0)
    first_bag = up_neighs[clique_neigh_idx] ∪ ordering[clique_neigh_idx]
    decomp = Graph(1)
    bags = [first_bag]
    tw = length(first_bag) - 1

    for i in clique_neigh_idx-1:-1:1
        neigh = up_neighs[i]

        # find a bag all neighbors are in
        old_bag_idx = 0
        for (j,bag) in enumerate(bags)
            if neigh ⊂ bag
                old_bag_idx = j
                break
            end
        end
        (old_bag_idx == 0) && (old_bag_idx = 1) # no old_bag was found: just connect to the first_bag

        # create new node
        add_vertex!(decomp)
        new_bag = [neigh; ordering[i]]
        push!(bags, new_bag)

        # update treewidth
        tw = max(tw, length(new_bag)-1)

        # add edge to decomposition
        LightGraphs.add_edge!(decomp, old_bag_idx, nv(decomp))

     end
     return tw, decomp, bags
end

"""
    is_tree_decomposition(G, tree, bags)

Checks if `(tree, bags)` forms a tree decomposition of `G`.
"""
function is_tree_decomposition(G, tree, bags)
    # 1. check union property
    if sort(∪(bags...)) != collect(1:nv(G))
        @warn ("Union of bags is not equal to union of vertices")
        return false
    end

    # 2. check edge property
    for edge in edges(G)
        edge_found = false
        e = (edge.src, edge.dst)
        for B in bags
            edge_found = edge_found | (e ⊂ B)
        end
        if ! edge_found
            @warn("Edge $e not found in any bag")
            return false
        end
    end

    # 3. check subtree property
    subgraphs = [Int[] for i in 1:nv(G)] # subgraph[i] are the bags that contain `i`
    for (i,b) in enumerate(bags)
        for j in b
            push!(subgraphs[j], i)
        end
    end

    for (v, s) in enumerate(subgraphs)
        if length(s) > 0
            subtree, _ = induced_subgraph(tree, s)
            if ! is_connected(subtree)
                @warn("Subgraph for vertex $v not connected")
                return false
            end
        end
    end
    return true
end

"""
    contraction_order(H::Graph, edges::Vector{NTuple{N, Int}})

Constructs a near optimal contracting order of the line_graph `H` with
associated edges `edges` by calling `tree_decomposition`,
following Theorem 4.6 of [Markov & Shi, Simulating quantum computation by contracting tensor networks](https://arxiv.org/abs/quant-ph/0511069.)
"""
function contraction_order(H::Graph, edges::Vector{NTuple{3,Int}}) where N
    (length(edges) == nv(H)) || error("Invalid list of edges for `H`; the length of `edges` must equal the number of vertices of `H`")
    tw, tree, bags = tree_decomposition(H)
    contr_order = NTuple{3,Int}[]
    degrees = length.(tree.fadjlist)
    while maximum(degrees) > 0
        leaves_idx = findall(degrees .== 1)
        leaf_idx = leaves_idx[argmin(length.(bags[leaves_idx]))]
        leaf_bag = bags[leaf_idx]
        neigh_idx = tree.fadjlist[leaf_idx][1] # only neighbor of leaf
        neigh_bag = bags[neigh_idx]
        diff_bag = setdiff(leaf_bag, neigh_bag)
        for i in diff_bag
            push!(contr_order, edges[i])
        end

        # update tree and bags
        rem_vertex!(tree, leaf_idx)
        moved_bag = pop!(bags)
        if leaf_idx ≤ nv(tree) # if removed other than last vertex
            bags[leaf_idx] = moved_bag
        end
        degrees = length.(tree.fadjlist)
    end

    # check that only one bag left
    @assert length(bags) == 1
    for i in bags[1]
        push!(contr_order, edges[i])
    end
    contr_order
end

"""
    contraction_order(net)

Return a near-optimal contraction order of the edges of `net`
"""
function contraction_order(net::TensorNetwork)
    _, edge_idx = network_graph(net)
    H, edges = line_graph(net)
    con_order = contraction_order(H, edges)

    # trace-like contractions where not taken into account;
    # move them to the beginning
    auto_con = NTuple{3, Int}[]
    for i in 1:length(net.tensors)
        if (i, i) in keys(edge_idx)
            for k in edge_idx[(i,i)]
                push!(auto_con, (i,i,k))
            end
        end
    end

    return [auto_con; con_order]
end

"""
    optimize_contraction_order!(net)

Optimize the contraction order for a TensorNetwork with no open legs.
The original paper [Markov & Shi, Simulating quantum computation by contracting tensor networks](https://arxiv.org/abs/quant-ph/0511069.)
considers circuits with a product state input and projection onto an
product state as output; as in the following example:

1 □—————————————————□———————————□
                    |
2 □———————————□—————□—————□—————□
              |           |
3 □—————□———————————————————————□
        |     |           |
4 □—————□—————□———————————□—————□

We see a great time saving also for MPS type circuits, **as long as no leg
remains uncontracted** (for example, the output is projected into another MPS).
The size of all legs of the original tensor should be roughly the same for
getting good performance (no different orders of magnitude)

For TensorNetwork with open legs this algorithm is unlikely to provide good
results, since it works by keeping the dimension of the contracted tensors as
small as possible (and this proves to be counterproductive in a network with open legs).
"""
function optimize_contraction_order!(net::TensorNetwork)
    (length(net.openidx) == 0) || @warn("For TensorNetworks with open indices the treewidth algorithm is unlikely to optimize performance")
    new_order = contraction_order(net)
    perm = [t[3] for t in new_order]
    net.contractions = net.contractions[perm]
    nothing
end
