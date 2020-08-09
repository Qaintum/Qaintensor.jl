using Test
using TestSetExtensions
using LightGraphs
using Qaintensor
using Qaintensor: random_graph, local_circuit_graph,
        tree_decomposition, is_tree_decomposition, ⊂,
        interaction_graph, lacking_for_clique_neigh, rem_vertex_fill!,
        min_fill_ordering, triangulation, contraction_order, contract
using Qaintessent

# for extensive check of the random tests increase the following parameter
samples_per_test = 1

Base.copy(net::TensorNetwork) = TensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))

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
where each tensor is a 2x2 random matrix.
"""
function test_setup()
    A = rand(ComplexF64, 2, 2)

    tensors = Tensor.([copy(A) for i in 1:4])
    contractions = Summation.([[1=>2, 2=>1],
                               [2=>2, 3=>1],
                               [3=>2, 4=>1],
                               [4=>2, 1=>1]])
    openidx = Pair[]
    TN0 = TensorNetwork(tensors, contractions, openidx)
end


@testset ExtendedTestSet "subset function" begin
    A = [1,2,3]
    B = [1,2,3,4]

    @test A ⊂ B
    @test !(B ⊂ A)

    # works for all iterable objects
    @test (1,2) ⊂ A
    @test A ⊂ (1,2,3,4)
    @test 1 ⊂ A
    @test 1:3 ⊂ A
    @test "ba" ⊂ "abc"

    # doesn't work for non-iterable objects
    @test_throws MethodError (XGate() ⊂ XGate())
end

@testset ExtendedTestSet "random_graph" begin
    Nn = 10
    Ne = 20
    G = random_graph(Nn, Ne)
    @test nv(G) == Nn
    @test ne(G) == Ne

    Ne = 46
    @test_throws ErrorException("Number of edges must be smaller or equal than N(N-1)/2, with N the number of vertices") random_graph(Nn, Ne)

end

@testset ExtendedTestSet "local_circuit_graph" begin
    N = 4
    G = Graph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    add_edge!(G, 3, 4)
    @test G == local_circuit_graph(4,2)

    add_edge!(G, 1, 3)
    add_edge!(G, 2, 4)
    @test G == local_circuit_graph(4,3)


    add_edge!(G, 1, 4)
    @test G == local_circuit_graph(4,4)
end

@testset ExtendedTestSet "network_graph" begin
    TN0 = test_setup()
    G, _ = network_graph(TN0)
    @test (nv(G) == 4) & prod(degree(G) .== 2) # the graph is a square

    # test error throw for contractions of more than two legs
    TN = copy(TN0)
    TN.contractions = [Summation([1=>2, 2=>1, 2=>2, 3=>1]),
                       Summation([3=>2, 4=>1, 4=>2, 1=>1])]
    @test_throws ErrorException("Contractions of more than 2 tensors not supported") network_graph(TN)
end

@testset ExtendedTestSet "line_graph" begin
    TN0 = test_setup()
    G, _ = network_graph(TN0)

    # test line_graph
    LG, _ = line_graph(G)
    @test (nv(LG) == 4) & prod(degree(LG) .== 2) # in this case G and LG are isomorphic

    # test line_graph for tensor networks
    # for tensor networks with only one leg between each pair of tensors
    # the result should be the same
    LG0, _ = line_graph(TN0)
    @test (nv(LG0) == 4) & prod(degree(LG0) .== 2)

    ## Random TN line_graph
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

    # Warning for tensor networks with open legs
    TN = copy(TN0)
    c = pop!(TN.contractions)
    push!(TN.openidx, c.idx[1])
    push!(TN.openidx, c.idx[2])
    @test_logs  (:warn, "All open indices are disregarded") line_graph(TN)
end

@testset ExtendedTestSet "interaction_graph" begin
    ## Interaction graph:
    # Interaction graph of the qft circuit is a complete graph
    N = 10
    cgc = qft_circuit(N)
    G = interaction_graph(cgc)
    @test G == complete_graph(N)
end

# Tree decomposition subroutines
@testset ExtendedTestSet "lacking_for_clique_neigh" begin
    G = complete_graph(5)
    @test (0,[]) == lacking_for_clique_neigh(G,1)
    rem_edge!(G,2,3)
    @test (1,[(2,3)]) == lacking_for_clique_neigh(G,1)
end

@testset ExtendedTestSet "rem_vertex_fill!" begin
    ## `rem_vertex_fill!`: eliminates a vertexs in the graph and fills
    # their neighborhood
    G = complete_graph(5)
    rem_edge!(G,2,3)
    ordering = Int[]
    vertex_label = [1, 2, 3, 4, 5]
    rem_vertex_fill!(G, 1, [(2,3)], ordering, vertex_label)

    @test G == complete_graph(4)
    @test ordering == [1]
    @test vertex_label == [5, 2, 3, 4]
end

@testset ExtendedTestSet "triangulation" begin
    ## `triangulation`: from a graph and an ordering get a chordal completion
    # (i. e. for every n-cycle (n>3) there is a 3-cycle that traverses
    # only three of the original nodes)
    # since enumerating the cycles on a graph grows faster than
    # exponentially in the number of edges, the parameters must be kept small
    for i in 1:samples_per_test
        Nn = rand(4:8)
        Ne = rand(Nn:(Nn*(Nn-1))÷2)
        G = random_graph(Nn, Ne)
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

@testset ExtendedTestSet "is_tree_decomposition" begin


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
        Ne = rand(3*Nn:Nn*(Nn-1)÷2)
        G = random_graph(Nn, Ne)
        tw, tree, bags = tree_decomposition(G)
        @test is_tree_decomposition(G, tree, bags)
    end

    # validate tree decomposition for line graph of random TN
    nodes_and_legs = [(10,10), (10,20), (10,50), (30,60)]
    for (Nn, Ne) in nodes_and_legs
        for i in 1:samples_per_test
            TN = random_TN(Nn, Ne)
            H, _ = line_graph(TN)
            tw, tree, bags = tree_decomposition(H)
            @test is_tree_decomposition(H, tree, bags)
        end
    end
end

@testset ExtendedTestSet "contraction_order" begin
    TN0 = test_setup()
    ## Contraction order:
    TN = copy(TN0)
    H, edges = line_graph(TN)
    order = contraction_order(H, edges)

    # Check that collecting third entries leads to a permutation
    @test sort([e[3] for e in order]) == [1:length(edges)...]
    # Check that if e = (i,j,k), contraction k is between tensors i and j
    @test prod(Set((TN.contractions[k].idx[1].first, TN.contractions[k].idx[2].first)) == Set((i,j))
                for (i, j, k) in edges)

    # If not same number of edges as vertices of H, throw error
    pop!(edges)
    @test_throws ErrorException("Invalid list of edges for `H`; the length of `edges` must equal the number of vertices of `H`") contraction_order(H, edges)
end

@testset ExtendedTestSet "optimize_contraction_order!" begin
    ## optimize_contraction_order!
    # Test that the tensor network is essentially the same after `optimize_contration_order!`
    TN0 = test_setup()
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
    @test_logs  (:warn, "For TensorNetworks with open indices the treewidth algorithm is unlikely to optimize performance") (:warn, "All open indices are disregarded") optimize_contraction_order!(TN)

    # Test on random TN
    nodes_and_legs = [(10,10), (10,20), (10,50), (20,60)]
    for (Nn, Ne) in nodes_and_legs
        for i in 1:samples_per_test
            net0 = random_TN(Nn, Ne)
            net = copy(net0)
            optimize_contraction_order!(net)
            @test net.tensors == net0.tensors
            @test net.openidx == net0.openidx
            @test Set(net.contractions) == Set(net0.contractions)
        end
    end

    # Test on expectation-valued MPS
    # TODO:
    # BEGIN DELETE: this code is copied from the `mps` branch; delete when merged

    function ClosedMPS(T::AbstractVector{Tensor})
        l = length(T)
        @assert ndims(T[1]) == 2
        for i in 2:l-1
            @assert ndims(T[i]) == 3
        end
         @assert ndims(T[l]) == 2

        contractions = [Summation([1 => 2, 2 => 1]); [Summation([i => 3,i+1 => 1]) for i in 2:l-1]]
        openidx = reverse([1 => 1; [i => 2 for i in 2:l]])
        tn = TensorNetwork(T, contractions, openidx)
        return tn
    end


    function shift_summation(S::Summation, step)
       return Summation([S.idx[i].first + step => S.idx[i].second for i in 1:2])
    end

    # END DELETE

    Base.ndims(T::Tensor) = ndims(T.data)
    Base.copy(net::TensorNetwork) = TensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))
    crand(dims...) = rand(ComplexF64, dims...)

    # generate expectation value tensor network
    """ Compute the expectation value of a random MPS when run through circuit `cgc`"""
    function expectation_value(cgc::CircuitGateChain{N}; is_decompose = false) where N

        tensors = Tensor.([crand(2,2), [crand(2,2,2) for i in 2:N-1]..., crand(2,2)])
        T0 = ClosedMPS(tensors)

        T = copy(T0)
        tensor_circuit!(T, cgc, is_decompose = is_decompose)

        # measure
        T.contractions = [T.contractions; shift_summation.(T0.contractions, length(T.tensors))]
        for i in 1:N
            push!(T.tensors, T0.tensors[N+1-i])
            push!(T.contractions, Summation([T.openidx[end], (length(T.tensors) => T0.openidx[N+1-i].second)]))
            pop!(T.openidx)
        end
        T
    end

    N = 4
    cgc = qft_circuit(N)
    for is_decompose in [true, false]
        net0 = expectation_value(cgc, is_decompose = is_decompose);
        net = copy(net0)
        optimize_contraction_order!(net)
        @test Set(net.contractions) == Set(net0.contractions)
        @test contract(net) ≈ contract(net0)
    end

end
