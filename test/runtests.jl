using Test
using TestSetExtensions
import Pkg

@testset "All the tests" begin
    @includetests ARGS
end
