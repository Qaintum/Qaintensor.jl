using Test
using TestSetExtensions
using Qaintensor
using Qaintensor: tree_decomposition, is_tree_decomposition, ⊂,
        interaction_graph, lacking_for_clique_neigh, rem_vertex_fill!,
        min_fill_ordering, triangulation
using LightGraphs

# for extensive check of the random tests change the following parameter
samples_per_test = 1


Base.copy(net::TensorNetwork) = TensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))

"""
    random_graph(Nn, Ne)

Random graph with Nn nodes and at most Ne edges (multiedges are not supported)
"""
function random_graph(Nn, Ne)
    G = Graph(Nn)
    for j in 1:Ne
        n1 = rand(1:Nn-1)
        n2 = rand(n1+1:Nn)
        add_edge!(G, n1, n2)
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
    random_TN(Nn, Ne)

Random TensorNetwork with Nn nodes and Ne legs
"""
function random_TN(Nn, Ne)
    nlegs = zeros(Int, Nn)
    contractions = Summation[]
    for j in 1:Ne
        n1 = rand(1:Nn-1)
        n2 = rand(n1+1:Nn)
        nlegs[n1] += 1
        nlegs[n2] += 1
        push!(contractions, Summation([n1 => nlegs[n1], n2 => nlegs[n2]]))
    end
    tensors = Tensor.([n > 0 ? rand(fill(2, n)...) : rand(1) for n in nlegs])
    return TensorNetwork(tensors, contractions, Pair{Integer,Integer}[])
end

"""
We will test a tensor network of the following form
    1 □—————□ 4
      |     |
    2 □—————□ 3
where each tensor is a 2x2 identity matrix.
"""
A = rand(ComplexF64, 2, 2)

tensors = Tensor.([copy(A) for i in 1:4])
contractions = Summation.([[1=>2, 2=>1],
                           [2=>2, 3=>1],
                           [3=>2, 4=>1],
                           [4=>2, 1=>1]])
openidx = Pair[]
TN0 = TensorNetwork(tensors, contractions, openidx)


@testset ExtendedTestSet "subset function" begin
    A = [1,2,3]
    B = [1,2,3,4]

    @test A ⊂ B
    @test !(B ⊂ A)
end

@testset ExtendedTestSet "network_graph" begin
    G, _ = network_graph(TN0)
    @test prod(length.(G.fadjlist) .== 2) # all vertices have 2 neighbors

    LG, _ = line_graph(TN0)
    @test prod(length.(LG.fadjlist) .== 2) # in this case G and LG are isomorphic

    # test error throw for contractions of more than two legs
    TN = copy(TN0)
    TN.contractions = [Summation([1=>2, 2=>1, 2=>2, 3=>1]),
                       Summation([3=>2, 4=>1, 4=>2, 1=>1])]
    @test_throws ErrorException network_graph(TN)

    ## Random TN:
    # check that the nodeinfo is composed of the tuples `(i,j,k)`, where
    # `i` and `j` are the tensors connected by the k-th sumation of the TN
    for i in 1:samples_per_test
        TN = random_TN(10,20)
        LG, nodeinfo = line_graph(TN)
        net_cont = Tuple{Int, Int, Int}[]
        for (k, con) in enumerate(TN.contractions)
            (i,j) = (con.idx[1].first, con.idx[2].first)
            i < j ? nothing : (i,j) = (j,i)
            push!(net_cont, (i,j,k))
        end
        @test Set(nodeinfo) == Set(net_cont)
    end

    ## Interaction graph:
    # Interaction graph of the qft circuit is a complete graph
    N = 10
    cgc = qft_circuit(N)
    G = interaction_graph(cgc)
    @test G == complete_graph(N)
end


@testset ExtendedTestSet "tree decomposition subroutines" begin

    ## test `lacking_for_clique_neigh` function
    G = complete_graph(5)
    @test (0,[]) == lacking_for_clique_neigh(G,1)
    rem_edge!(G,2,3)
    @test (1,[(2,3)]) == lacking_for_clique_neigh(G,1)

    ##  `rem_vertex_fill!`: that eliminates vertices in the graph and fills
    # their neighborhood
    ordering = Int[]
    vertex_label = [1, 2, 3, 4, 5]
    rem_vertex_fill!(G, 1, [(2,3)], ordering, vertex_label)

    @test G == complete_graph(4)
    @test ordering == [1]
    @test vertex_label == [5, 2, 3, 4]

    ## `triangulation`: from a graph and an ordering we get a chordal completion
    # (i. e. for every n-cycle (n>3) there is a 3-cycle that traverses
    # only three of the original nodes)
    # since enumerating the cycles on a graph grows faster than
    # exponentially in the number of edges, the parameters must be kept small
    for i in 1:samples_per_test
        Nn = rand(4:8)
        Ni = rand(2:2:Nn-1)
        G = random_regular_graph(Nn, Ni)
        ordering = min_fill_ordering(G)
        H = triangulation(G, ordering)

        dg = DiGraph(H) # cycle enumerating only works on digraphs
        cycles = simplecycles(dg)
        length_3_cycles = cycles[length.(cycles) .== 3]
        success = true
        for cycle in cycles[length.(cycles) .> 3]
            success = success & any(map(x -> x ⊂ cycle, length_3_cycles))
        end
        @test success
    end
end

@testset ExtendedTestSet "tree decomposition" begin
    ## tree decomposition of a complete graph
    # A complete graph has itself as its only chordal completion.
    # Since the min_fill_ordering works by constructing a chordal completion,
    # and for complete graphs there is only one, the approximated treewidth is exact.
    for n in [10, 25, 50]
        G = complete_graph(n)
        tw, _ = tree_decomposition(G)
        @test tw == n-1
    end

    # test treewidth of local circuit graph (heuristic should generate an optimal tree decomposition)
    for n in 2:5
        IG = local_circuit_graph(10, n)
        tw, _ = tree_decomposition(IG)
        @test tw == n-1
    end
end

@testset ExtendedTestSet "tree decomposition validation" begin


    # We consider the following graph and tree decomposition
    #    5
    #  /   \            tree                   [2, 3, 4]
    # 4 ——— 3            -->                    /     \
    # |     |       decomposition       [2, 4, 1]     [3, 4, 5]
    # 1 ——— 2


    # `ìs_tree_decomposition`: validates tree decompositions
    G = Graph(5)
    add_edge!(G, 1, 2)
    add_edge!(G, 1, 4)
    add_edge!(G, 2, 3)
    add_edge!(G, 4, 3)
    add_edge!(G, 5, 3)
    add_edge!(G, 4, 5)

    tree = Graph(3)
    add_edge!(tree, 1, 2)
    add_edge!(tree, 1, 3)

    bags = [[2, 3, 4],
            [2, 4, 1],
            [3, 4, 5]]

    @test is_tree_decomposition(G, tree, bags)

    # removing node 1 of the second bag leads to an invalid
    # tree decomposition (not all vertices present)
    bags = [[2, 3, 4],
            [2, 4],
            [3, 4, 5]]

    @test ! (@test_logs (:warn, "Union of bags is not equal to union of vertices") is_tree_decomposition(G, tree, bags))


    # deleting node 2 of the first bag leads to an invalid
    # tree decomposition (edge (2, 3) not contained in any bag)
    bags = [[3, 4],
            [2, 4, 1],
            [3, 4, 5]]
    e = (2,3)
    @test ! (@test_logs (:warn, "Edge $e not found in any bag") is_tree_decomposition(G, tree, bags))

    # removing node 4 of the first bag leads to an invalid
    # tree decomposition (subgraph for vertex 4 not connected)
    bags = [[2, 3],
            [2, 4, 1],
            [3, 4, 5]]

    @test ! (@test_logs (:warn, "Subgraph for vertex 4 not connected") is_tree_decomposition(G, tree, bags))


    # validate tree decomposition algorithm on random graphs
    for i in 1:samples_per_test
        Nn = rand(20:50)
        Ne = rand(2*Nn:10*Nn)
        G = random_graph(Nn, Ne)
        tw, tree, bags = tree_decomposition(G)
        @test is_tree_decomposition(G, tree, bags)
    end

    # on random TN
    nodes_and_legs = [(20,5), (10,10), (10,20), (30,60)]
    for (Nn, Ne) in nodes_and_legs
        for i in 1:samples_per_test
            TN = random_TN(Nn, Ne)
            H, _ = line_graph(TN)
            tw, tree, bags = tree_decomposition(H)
            @test is_tree_decomposition(H, tree, bags)
        end
    end

end

@testset ExtendedTestSet "optimize contraction" begin
    # Test that the tensor network is essentially the same
    TN = copy(TN0)
    optimize_contraction_order!(TN)
    @test TN.tensors == TN0.tensors
    @test TN.openidx == TN0.openidx
    @test Set(TN.contractions) == Set(TN0.contractions)
    @test Qaintensor.contract(TN) ≈ Qaintensor.contract(TN0)

    # Warning for tensor networks with open legs
    TN = copy(TN0)
    c = pop!(TN.contractions)
    push!(TN.openidx, c.idx[1])
    push!(TN.openidx, c.idx[2])

    @test_logs  (:warn, "For TensorNetworks with open indices the treewidth algorithm is unlikely to optimize performance.") optimize_contraction_order!(TN)
end
