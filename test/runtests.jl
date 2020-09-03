using Test
using TestSetExtensions
import Pkg

Pkg.add(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintessent.jl", rev="flip_qubit_order"))

@testset "All the tests" begin
    @includetests ARGS
end
