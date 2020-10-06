# MPO Representation

As a quantum state can be converted into MPS states, an arbitrary quantum operation can similarly be converted into a Matrix-Product Operator (MPO) form. MPO objects can be created from an AbtractMatrix or a given CircuitGate object.

```@meta
CurrentModule = Qaintensor
```

```@docs
MPO
MPO(m::AbstractMatrix)
MPO(cg::CircuitGate)
MPO(cg::AbstractCircuitGate)
```

MPO objects can be applied to TensorNetwork objects via the `apply_MPO` function prior to contraction, given a tuple of wires the MPO is applied to.

```@docs
apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where {M}
```
