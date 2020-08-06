using Test
using TestSetExtensions


@testset "All the tests" begin
    # @includetests ["test_helper"]
    @includetests ["test_mps"]
    # @includetests ARGS
end
