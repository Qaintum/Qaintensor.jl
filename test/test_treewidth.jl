using Test
using TestSetExtensions
using Qaintensor
using Qaintensor: tree_decomposition, is_tree_decomposition, ⊂,
        interaction_graph, lacking_for_clique_neigh, rem_vertex_fill!
using LightGraphs

Base.copy(net::TensorNetwork) = TensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))

function random_graph(Nn, Ne)
    G = Graph(Nn)
    for j in 1:Ne
        n1 = rand(1:Nn-1)
        n2 = rand(n1:Nn)
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

    # the interaction graph of the qft is the complete graph
    N = 10
    cgc = qft_circuit(N)
    G = interaction_graph(cgc)
    @test ne(G) == N*(N-1)/2
end


@testset ExtendedTestSet "tree decomposition" begin

    # test `lacking_for_clique_neigh` function
    G = complete_graph(5)
    @test (0,[]) == lacking_for_clique_neigh(G,1)
    rem_edge!(G,2,3)
    @test (1,[(2,3)]) == lacking_for_clique_neigh(G,1)

    # test rem_vertex_fill!,that eliminates vertices in the graph and fills
    # their neighborhood
    ordering = Int[]
    vertex_label = [1, 2, 3, 4, 5]
    rem_vertex_fill!(G, 1, [(2,3)], ordering, vertex_label)

    @test G == complete_graph(4)
    @test ordering == [1]
    @test vertex_label == [5, 2, 3, 4]

    # test for complete graph
    # A complete graph has itself as its only chordal completion.
    # Since the min_fill_ordering works by constructing a chordal completion,
    # and for complete graphs there is only one, the approximated treewidth is tight.
    for n in [10, 25, 50]
        G = complete_graph(n)
        tw, _ = tree_decomposition(G)
        @test tw == n-1
    end

    # test for validity of the tree decomposition
    for i in 1:5
        Nn = rand(20:50)
        Ne = rand(2*Nn:10*Nn)
        G = random_graph(Nn, Ne)
        tw, tree, bags = tree_decomposition(G)
        @test is_tree_decomposition(G, tree, bags)
    end

    # test treewidth of local circuit graph (heuristic should generate an optimal tree decomposition)
    for n in 2:5
        IG = local_circuit_graph(10, n)
        tw, _ = tree_decomposition(IG)
        @test tw == n-1
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

    @test_logs  (:warn, "For TensorNetworks with open indices the treewidth algorithm is unlikely to optimize performance.")
end
