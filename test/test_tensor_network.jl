using Test
using TestSetExtensions
using Qaintensor
using Qaintessent

Base.copy(net::TensorNetwork) = TensorNetwork(copy(net.tensors), copy(net.contractions), copy(net.openidx))

@testset ExtendedTestSet "contraction" begin

    """ We will test a tensor network of the following form
        1 □—————□ 4
          |     |                                 
        2 □—————□ 3
        where each tensor is a 2x2 identity matrix."""
    
    A = [1. 0; 0 1]

    tensors = Tensor.([copy(A) for i in 1:4])
    contractions = Summation.([[1=>2, 2=>1],
                               [2=>2, 3=>1],
                               [3=>2, 4=>1],
                               [4=>2, 1=>1]])
    openidx = Pair[]
    TN0 = TensorNetwork(tensors, contractions, openidx) 
    
    # Check contraction
    @test contract(TN0)[1] == 2 # should equal trace of identity 

    # Check errors
    TN = copy(TN0)
    TN.tensors[1] = Tensor([1,1]) # replace first tensor by another with wrong number of dimensions
    @test_throws DimensionMismatch contract(TN) 

    TN = copy(TN0)
    TN.tensors[1] = Tensor(kron(A, A)) # replace first tensor by another with wrong size
    @test_throws DimensionMismatch contract(TN) 

    TN = copy(TN0)
    push!(TN.openidx, 1=>1) # add as openidx a leg that is supposed to be contracted
    @test_throws DimensionMismatch contract(TN) 

    TN = copy(TN0)
    pop!(TN.contractions) # remove a contraction, leaving ambiguous indices
    @test_throws DimensionMismatch contract(TN) 
end