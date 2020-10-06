# MPS Representation

One of the most common specialized tensor networks is the Matrix-Product State (MPS). This can be constructed from a given quantum state in state vector form via a series of SVD contractions.

```@meta
CurrentModule = Qaintensor
```
Qaintensor.jl currently supports the generation of a general `MPS` objet from a given quantum state.
```@docs
MPS(ψ::AbstractVector{ComplexF64})
```

Specialized `MPS` forms, such as `OpenMPS` or `ClosedMPS` are also supported.

```@docs
OpenMPS(T::AbstractVector{Tensor})
ClosedMPS(T::AbstractVector{Tensor})
```

MPS objects can be contracted via the `contract` function (see [TensorNetwork Contractions](@ref)).
