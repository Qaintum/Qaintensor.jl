using Test
using TestSetExtensions
import Pkg

Pkg.add(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintessent.jl"))

@testset "All the tests" begin
    @includetests ARGS
end
