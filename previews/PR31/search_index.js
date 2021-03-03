var documenterSearchIndex = {"docs":
[{"location":"tensors/#Tensor-Networks","page":"Tensor Networks","title":"Tensor Networks","text":"","category":"section"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"The a tensor network consists of multiple tensors connected by bonds. The contraction of these bonds determines the final output of the tensor network. Quantum Circuits are a specialized subset of tensor networks.","category":"page"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"CurrentModule = Qaintensor","category":"page"},{"location":"tensors/#Tensors","page":"Tensor Networks","title":"Tensors","text":"","category":"section"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"The basic building block of any GeneralTensorNetwork is a Tensor. These are constructed from an arbitrary tensor or high-order AbstractMatrix.","category":"page"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"Tensor","category":"page"},{"location":"tensors/#Qaintensor.Tensor","page":"Tensor Networks","title":"Qaintensor.Tensor","text":"Tensor\n\nStores tensor data\n\n\n\n\n\n","category":"type"},{"location":"tensors/#GeneralTensorNetwork","page":"Tensor Networks","title":"GeneralTensorNetwork","text":"","category":"section"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"All tensor network objects are based off the abstract TensorNetwork struct. The most general form is the GeneralTensorNetwork. This object can be created by providing a list of Tensors objects, contractions sequences and open indices.","category":"page"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"GeneralTensorNetwork","category":"page"},{"location":"tensors/#Qaintensor.GeneralTensorNetwork","page":"Tensor Networks","title":"Qaintensor.GeneralTensorNetwork","text":"GeneralTensorNetwork  <: TensorNetwork\n\nGeneral Tensor network, consisting of tensors and contraction operations\n\n\n\n\n\n","category":"type"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"TensorNetwork objects can be contracted via the contract function.","category":"page"},{"location":"tensors/#TensorNetwork-Contractions","page":"Tensor Networks","title":"TensorNetwork Contractions","text":"","category":"section"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"TensorNetworks can be contracted via the contract function.","category":"page"},{"location":"tensors/","page":"Tensor Networks","title":"Tensor Networks","text":"contract(net::TensorNetwork; optimize=false)","category":"page"},{"location":"tensors/#Qaintensor.contract-Tuple{TensorNetwork}","page":"Tensor Networks","title":"Qaintensor.contract","text":"contract(net::TensorNetwork; optimize=false)\n\nFully contract a given TensorNetwork object.\n\n\n\n\n\n","category":"method"},{"location":"mps/#MPS-Representation","page":"MPS Representation","title":"MPS Representation","text":"","category":"section"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"One of the most common specialized tensor networks is the Matrix-Product State (MPS). This can be constructed from a given quantum state in state vector form via a series of SVD contractions.","category":"page"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"CurrentModule = Qaintensor","category":"page"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"Qaintensor.jl currently supports the generation of a general MPS objet from a given quantum state.","category":"page"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"MPS(ψ::AbstractVector{ComplexF64})","category":"page"},{"location":"mps/#Qaintensor.MPS-Tuple{AbstractArray{Complex{Float64},1}}","page":"MPS Representation","title":"Qaintensor.MPS","text":"MPS(ψ::AbstractVector{ComplexF64})\n\nConvert a vector ψ into a MPS reprsentation of it.\n\n\n\n\n\n","category":"method"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"Specialized MPS forms, such as OpenMPS or ClosedMPS are also supported.","category":"page"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"OpenMPS(T::AbstractVector{Tensor})\nClosedMPS(T::AbstractVector{Tensor})","category":"page"},{"location":"mps/#Qaintensor.OpenMPS-Tuple{AbstractArray{Tensor,1}}","page":"MPS Representation","title":"Qaintensor.OpenMPS","text":"OpenMPS(T::AbstractVector{Tensor})\n\nCreate a MPS formed by the tensors in T with open boundary conditions (that is, each tensor has 2 virtual legs and a physical one, including those on the boundaries).\n\n\n\n\n\n","category":"method"},{"location":"mps/#Qaintensor.ClosedMPS-Tuple{AbstractArray{Tensor,1}}","page":"MPS Representation","title":"Qaintensor.ClosedMPS","text":"ClosedMPS(T::AbstractVector{Tensor})\n\nCreate a MPS formed by the tensors  in T with closed boundary conditions (that is, each tensor has 2 virtual legs and a physical one, except those at the boundaries that have only one virtual leg).\n\n\n\n\n\n","category":"method"},{"location":"mps/","page":"MPS Representation","title":"MPS Representation","text":"MPS objects can be contracted via the contract function (see TensorNetwork Contractions).","category":"page"},{"location":"mpo/#MPO-Representation","page":"MPO Representation","title":"MPO Representation","text":"","category":"section"},{"location":"mpo/","page":"MPO Representation","title":"MPO Representation","text":"As a quantum state can be converted into MPS states, an arbitrary quantum operation can similarly be converted into a Matrix-Product Operator (MPO) form. MPO objects can be created from an AbtractMatrix or a given CircuitGate object.","category":"page"},{"location":"mpo/","page":"MPO Representation","title":"MPO Representation","text":"CurrentModule = Qaintensor","category":"page"},{"location":"mpo/","page":"MPO Representation","title":"MPO Representation","text":"MPO\nMPO(m::AbstractMatrix)\nMPO(cg::CircuitGate)\nMPO(cg::AbstractCircuitGate)","category":"page"},{"location":"mpo/#Qaintensor.MPO","page":"MPO Representation","title":"Qaintensor.MPO","text":"MPO(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer\n\nSubclassed TensorNetwork object in Matrix Product Operator(MPO) form.\n\n\n\n\n\n","category":"type"},{"location":"mpo/#Qaintensor.MPO-Tuple{AbstractArray{T,2} where T}","page":"MPO Representation","title":"Qaintensor.MPO","text":"MPO(m::AbstractMatrix)\n\nTransform an operator represented by matrix m into an MPO form.\n\n\n\n\n\n","category":"method"},{"location":"mpo/#Qaintensor.MPO-Tuple{Qaintessent.CircuitGate}","page":"MPO Representation","title":"Qaintensor.MPO","text":"MPO(cg::CircuitGate)\n\nTransform an operator represented by a gate cg into an MPO form.\n\n\n\n\n\n","category":"method"},{"location":"mpo/#Qaintensor.MPO-Tuple{Qaintessent.AbstractCircuitGate}","page":"MPO Representation","title":"Qaintensor.MPO","text":"MPO(cg::CircuitGate)\n\nTransform an operator represented by a gate cg into an MPO form.\n\n\n\n\n\n","category":"method"},{"location":"mpo/","page":"MPO Representation","title":"MPO Representation","text":"MPO objects can be applied to TensorNetwork objects via the apply_MPO function prior to contraction, given a tuple of wires the MPO is applied to.","category":"page"},{"location":"mpo/","page":"MPO Representation","title":"MPO Representation","text":"apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where {M}","category":"page"},{"location":"mpo/#Qaintensor.apply_MPO-Union{Tuple{M}, Tuple{TensorNetwork,MPO,Tuple{Vararg{var\"#s1\",M}} where var\"#s1\"<:Integer}} where M","page":"MPO Representation","title":"Qaintensor.apply_MPO","text":"apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where {M}\n\nGiven a state ψ  in a Tensor Network form and an operator mpo acting on M qudits, update the state by effectively applying mpo.\n\nArguments\n\niwire::NTuple{M, <:Integer}: qudits in which MPO acts. When the input is mpo::MPO, iwires must be sorted.\n\n\n\n\n\n","category":"method"},{"location":"#Qaintensor.jl-Documentation","page":"Home","title":"Qaintensor.jl Documentation","text":"","category":"section"},{"location":"#Table-of-Contents","page":"Home","title":"Table of Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\", \"tensors.md\", \"mps.md\", \"mpo.md\"]","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Qaintensor","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Qaintensor.jl is an extension to the  digital quantum circuit toolbox and simulator Qaintessent.jl. This library allows the conversion of Circuit/CircuitGateChain objects created in Qaintessent.jl into TensorNetwork objects. (See Tensors for details regarding conversion and the creation of general TensorNetworks). Qaintensor.jl currently supports MPS and MPO states, see MPS Representation and MPO Representation.","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
