import Pkg

try
    Pgk.rm(Pkg.PackageSpec(url="https://github.com/Qaintum/Qaintmodels.jl"))
catch
end


Pkg.add(url="https://github.com/Qaintum/Qaintmodels.jl")
