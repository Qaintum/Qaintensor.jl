"""
    MPS(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer

Subclassed TensorNetwork object in Matrix-Product-State(MPS) form. Tensor objects must have each have a maximum of three legs and are ordered
such that each Tensor object only connects to the Tensor object before and after with 1 leg each.
"""
mutable struct MPS <: TensorNetwork
    # list of tensors
    tensors::AbstractVector{Tensor}
    # contractions, specified as list of summations
    contractions::AbstractVector{Summation}
    # ordered "open" (uncontracted) indices (list of tensor and leg indices)
    openidx::AbstractVector{Pair{Integer,Integer}}

    function MPS(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer
        # Checks to ensure MPS is in correct form, ensure that contractions are ordered correctly
        for (i, id) in enumerate(contractions)
            @assert length(id.idx) == 2
            (id.idx[1].second == 3) || error("Tensor objects first leg must contract with last leg of previous Tensor object. ")
            (id.idx[2].second == 1) || error("Tensor objects last leg must contract with first leg of next Tensor object. ")
        end

        for tensor in tensors
            length(size(tensor)) <= 3 || length(size(tensor)) >= 2 || error("Each Tensor object in MPS form can only have 2 or 3 legs")
        end

        new(tensors, contractions, openidx)
    end
end

""""
    check_mps(mps::MPS)

Run checks on MPS object to ensure that structure has not been changed.
"""

function check_mps(mps::MPS)
    for tensor in mps.tensors
        length(size(tensor)) in [2,3] || error("Each Tensor object in MPS form can only have 2 or 3 legs")
    end

    for (i, id) in enumerate(mps.contractions)
        @assert length(id.idx) == 2
        tensor_size = length(size(mps.tensors[id.idx[1].first]))
        (id.idx[1].second == tensor_size) || error("Tensor objects first leg must contract with last leg of previous Tensor object")
        (id.idx[2].second == 1) || error("Tensor objects last leg must contract with first leg of next Tensor object")
    end
end

"""
    MPS(ψ::AbstractVector{ComplexF64})

Convert a vector `ψ` into a MPS reprsentation of it.
"""
function MPS(ψ::AbstractVector{ComplexF64})
    is_power_two(length(ψ)) || error("Input state must have length 2^N")
    M = Integer(log(2, length(ψ)))
    tensors = Tensor[]
    contractions = Summation[]
    openidx = Pair{Integer,Integer}[]

    ψ = reshape(ψ, (2, 2^(M-1)))
    S, V, D = svd(ψ)

    push!(tensors, Tensor(S))
    pushfirst!(openidx, 1=>1)
    lbond = length(V)
    ψ = diagm(V) * adjoint(D)
    lastbit = 2

    for bit in 2:M-1
        ψ = reshape(ψ, (lbond*2, 2^(M-bit)))
        U, S, V = svd(ψ)

        rbond = length(S)
        ψ = diagm(S) * adjoint(V)
        push!(tensors, Tensor(reshape(U, (lbond, 2, rbond))))
        push!(contractions, Summation([bit-1=>lastbit, bit=>1]))
        pushfirst!(openidx, bit=>2)
        lbond = rbond
        lastbit = 3
    end

    push!(tensors, Tensor(ψ))
    push!(contractions, Summation([M-1=>lastbit, M=>1]))
    pushfirst!(openidx, M=>2)

    MPS(tensors, contractions, openidx)
end


"""
    OpenMPS(T::AbstractVector{Tensor})

Create a MPS formed by the tensors in `T` with open boundary conditions
(that is, each tensor has 2 virtual legs and a physical one, including
those on the boundaries).
"""
function OpenMPS(T::AbstractVector{Tensor})
    l = length(T)
    for i in 1:l
        @assert ndims(T[i]) == 3
    end

    contractions = [Summation([i => 3,i+1 => 1]) for i in 1:l-1]
    openidx = reverse([1 => 1; [i => 2 for i in 1:l]; l => 3])
    tn = MPS(T, contractions, openidx)
    return tn
end

"""
    OpenMPS(T::Tensor, N::Integer)

Create a translational invariant MPS of length `N` formed by tensors `T` with open boundary conditions
(that is, each tensor has 2 virtual legs and a physical one, including
those on the boundaries).
"""
function OpenMPS(T::Tensor, N::Integer)
    #translational invariant MPS
    return OpenMPS(fill(T, N))
end

"""
    ClosedMPS(T::AbstractVector{Tensor})

Create a MPS formed by the tensors  in `T` with closed boundary conditions
(that is, each tensor has 2 virtual legs and a physical one, except those at the
boundaries that have only one virtual leg).
"""
function ClosedMPS(T::AbstractVector{Tensor})
    l = length(T)
    @assert ndims(T[1]) == 2
    for i in 2:l-1
        @assert ndims(T[i]) == 3
    end
     @assert ndims(T[l]) == 2

    contractions = [Summation([1 => 2, 2 => 1]); [Summation([i => 3,i+1 => 1]) for i in 2:l-1]]
    openidx = reverse([1 => 1; [i => 2 for i in 2:l]])
    tn = MPS(T, contractions, openidx)
    return tn
end

"""
    ClosedMPS(Tfirst::Tensor, Tmiddle::Tensor, Tend::Tensor, N::Integer)

Create a translational invariant MPS of length `N` formed by tensors
`Tfirst`- `Tmiddle`- ... - `Tmiddle`- `Tlast` with closed boundary conditions
(that is, each tensor has 2 virtual legs and a physical one, except those at the
boundaries that have only one virtual leg).
"""
function ClosedMPS(Tfirst::Tensor, Tmiddle::Tensor, Tend::Tensor, N::Integer)
    return ClosedMPS([Tfirst; fill(Tmiddle, N-2); Tend])
end

"""
    PeriodicMPS(T::AbstractVector{Tensor})

Create a MPS of formed by the tensors in `T` with periodic boundary conditions
(that is, the first and last tensors are joined trough a virtual leg).
"""
function PeriodicMPS(T::AbstractVector{Tensor})
    l=length(T)
    for i in 1:l
        @assert Qaintensor.ndims(T[i]) == 3
    end

    contractions = [[Summation([i => 3,i+1 => 1]) for i in 1:l-1]; Summation([l => 3, 1 => 1])]
    openidx = reverse([i => 2 for i in 1:l])
    tn = MPS(T, contractions, openidx)
    return tn
end
"""
    PeriodicMPS(T::AbstractVector{Tensor})

Create a MPS of length `N` formed by the tensors `T` with periodic boundary conditions
(that is, the first and last tensors are joined trough a virtual leg).
"""
function PeriodicMPS(T::Tensor, N::Integer)
    #translational invariant MPS
    return PeriodicMPS(fill(T, N))
end

"""
    contract_svd_mps(tn::MPS; er::Real=0.0)

Contract an MPS object with a maximum error of truncation `er` in each
individual contraction.
"""
function contract_svd_mps(tn::MPS; er::Real=0.0)

    er >= 0 || error("Error must be positive")

    lchain = length(tn.tensors)
    (! any([Summation([lchain=>3, 1=>1]) == s for s in tn.contractions])) && (! any([Summation([1=>1, lchain=>3]) == s for s in tn.contractions]))  || error("Function doesn't support periodic boundary conditions for now")
    tcontract = tn.tensors[1]
    for j in 2:lchain
        tcontract = contract_svd(tcontract, tn.tensors[j], (ndims(tcontract),1) , er=er)
    end
    return tcontract.data
end

Base.copy(net::MPS) = MPS(copy(net.tensors), copy(net.contractions), copy(net.openidx))
