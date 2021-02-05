import Pkg

try
    Pkg.rm(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintessent.jl"))
    Pgk.rm(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintmodels.jl"))
catch
end


Pkg.add(url="https://github.com/Qaintum/Qaintessent.jl")
Pkg.add(url="https://github.com/Qaintum/Qaintmodels.jl")
