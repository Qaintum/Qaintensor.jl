using Test
using TestSetExtensions
import Pkg

Pkg.add(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintessent.jl"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintmodels.jl"))

@testset "All the tests" begin
    @includetests ARGS
end
