using Test
using TestSetExtensions
import Pkg

try
    Pkg.instantiate()
catch e
    Pkg.add(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintessent.jl"))
end
@testset "All the tests" begin
    @includetests ARGS
end
