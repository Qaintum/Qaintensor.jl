"""
    MPS(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer

Subclassed TensorNetwork object in Matrix-Product-State(MPS) form. Tensor objects must have each have a maximum of three legs and are ordered
such that each Tensor object only connects to the Tensor object before and after with 1 leg each.
"""
struct MPS <: TensorNetwork
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
            (id.idx[1].second == 3) || "Tensor objects first leg must contract with last leg of previous Tensor object. "
            (id.idx[2].second == 1) || "Tensor objects last leg must contract with first leg of next Tensor object. "
        end

        for tensor in tensors
            length(size(tensor)) <= 3 || length(size(tensor)) >= 2 || "Each Tensor object in MPS form can only have 2 or 3 legs"
        end

        new(tensors, contractions, openidx)
    end
end

""""
    check_mps(mps::MPS)

runs checks on MPS object to ensure that structure has not been changed
"""

function check_mps(mps::MPS)
    for (i, id) in enumerate(mps.contractions)
        @assert length(id.idx) == 2
        (id.idx[1].second == 3) || "Tensor objects first leg must contract with last leg of previous Tensor object. "
        (id.idx[2].second == 1) || "Tensor objects last leg must contract with first leg of next Tensor object. "
    end

    for tensor in mps.tensors
        length(size(tensor)) <= 3 || length(size(tensor)) >= 2 || "Each Tensor object in MPS form can only have 2 or 3 legs"
    end
end

"""
    is_power_two(i::Integer)

resturns true if i is a power of 2, else false
"""
function is_power_two(i::Integer)
    i != 0 || return false
    return (i & (i - 1)) == 0
end


function MPS(ψ::AbstractVector{ComplexF64})
    is_power_two(length(ψ)) || error("Input state must have length 2^N")
    M = Integer(log(2, length(ψ)))
    # T = TensorNetwork([], [], [])
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

function OpenMPS(T::Tensor, N::Integer)
    #translational invariant MPS
    return OpenMPS(fill(T, N))
end

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

function ClosedMPS(Tfirst::Tensor, Tmiddle::Tensor, Tend::Tensor, N::Integer)
    return ClosedMPS([Tfirst; fill(Tmiddle, N-2); Tend])
end

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

function PeriodicMPS(T::Tensor, N::Integer)
    #translational invariant MPS
    return PeriodicMPS(fill(T, N))
end

"""
    contract(tn::MPS, er::Real)

tn: TensorNetwork. Must be provided in a MPS form, that is, tn.tensors have three legs,
    tn.contractions are of the form[(T_i, 3), (T_i+1,1)]
er: maximum error in the truncation done in an individual contraction
"""
function contract(tn::MPS; er::Real=0.0)

    lchain = length(tn.tensors)
    tcontract = tn.tensors[1]
    k = length(size(tcontract)) - 2
    for j in 2:lchain
        tcontract = contract_svd(tcontract, tn.tensors[j], (j+k,1); er=er)
    end
    return tcontract.data
end
