# Tensor Networks

The a tensor network consists of multiple tensors connected by bonds. The contraction of these bonds determines the final output of the tensor network. Quantum Circuits are a specialized subset of tensor networks.

```@meta
CurrentModule = Qaintensor
```

## Tensors
The basic building block of any [GeneralTensorNetwork](@ref) is a Tensor. These are constructed from an arbitrary tensor or high-order AbstractMatrix.

```@docs
Tensor
```

### GeneralTensorNetwork
All tensor network objects are based off the abstract TensorNetwork struct. The most general form is the GeneralTensorNetwork. This object can be created by providing a list of [Tensors](@ref) objects, contractions sequences and open indices.

```@docs
GeneralTensorNetwork
```
TensorNetwork objects can be contracted via the contract function.

### TensorNetwork Contractions
TensorNetworks can be contracted via the `contract` function.

```@docs
contract(net::TensorNetwork; optimize=false)
```
