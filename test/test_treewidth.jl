using Test
using TestSetExtensions
using Qaintensor
using Qaintensor: tree_decomposition, is_tree_decomposition
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


@testset ExtendedTestSet "network_graph" begin
    G, _ = network_graph(TN0)
    @test prod(length.(G.fadjlist) .== 2) # all vertices have 2 neighbors

    LG, _ = line_graph(TN0)
    @test prod(length.(LG.fadjlist) .== 2) # they are isomorphic
end


@testset ExtendedTestSet "tree decomposition" begin

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
    @test contract(TN) ≈ contract(TN0)

    # Warning for tensor networks with open legs
    TN = copy(TN0)
    c = pop!(TN.contractions)
    push!(TN.openidx, c.idx[1])
    push!(TN.openidx, c.idx[2])

    @test_logs  (:warn, "For TensorNetworks with open indices the treewidth algorithm is unlikely to optimize performance.")
end
