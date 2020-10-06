using LinearAlgebra

"""
    MPO(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer

Subclassed TensorNetwork object in Matrix Product Operator(MPO) form.
"""
mutable struct MPO <: TensorNetwork
    # list of tensors
    tensors::AbstractVector{Tensor}
    # contractions, specified as list of summations
    contractions::AbstractVector{Summation}
    # ordered "open" (uncontracted) indices (list of tensor and leg indices)
    openidx::AbstractVector{Pair{Integer,Integer}}

    function MPO(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer
        # Checks to ensure MPS is in correct form, ensure that contractions are ordered correctly
        error("Direct conversion to MPS form is not support, please construct MPO from matrix or CircuitGate objects")
    end


    """@docs
        MPO(m::AbstractMatrix)

    Transform an operator represented by matrix `m` into an `MPO` form.
    """
    function MPO(m::AbstractMatrix)

        # TODO: support general "qudits"
        d = 2
        @assert size(m)[1] == size(m)[2]

        M = Int(log2(size(m)[1]))
        M ≥ 1 || error("Need at least one qubit to act on.")

        t = Tensor[]
        con = Summation[]
        openidx = Pair{Integer,Integer}[]

        if M > 1
            m = reshape(m, fill(2, 2M)...)
            bond_dim = 1
            dims = Integer[]

            for i in 1:M
                push!(dims, i)
                push!(dims, i + M)
            end
            m = permutedims(m, dims)
            m = reshape(m, (4*bond_dim, :))

            U, S, V = svd(Array(m))
            bond_dim = size(S)[1]

            m = diagm(S) * adjoint(V)
            push!(t, Tensor(reshape(U, (2, 2, bond_dim))))
            push!(con, Summation([1 => 3, 2 => 1]))

            for i in 2:M-1
                m = reshape(m, (bond_dim * 4, :))
                U, S, V = svd(Array(m))
                m = diagm(S) * adjoint(V)
                new_bond_dim = size(S)[1]

                push!(t, Tensor(reshape(U, (bond_dim, 2, 2, new_bond_dim))))
                push!(con, Summation([i => 4, i+1 => 1]))
                bond_dim = new_bond_dim
            end
            push!(t, Tensor(reshape(m, (bond_dim, 2, 2))))

            #output dimensions
            for i in 1:M-1
                push!(openidx, M-i+1 => 2)
            end
            push!(openidx, 1 => 1)

            #input dimensions
            for i in 1:M-1
                push!(openidx, M-i+1 => 3)
            end
            push!(openidx, 1 => 2)
            else
            #one-qubit operator
            m = reshape(m, 2, 2)
            push!(t, Tensor(m))
            openidx = [1 => 2, 1 => 1]
        end
        new(t, con, openidx)
    end


    """@docs
        MPO(cg::AbstractGate)

    Transform an operator represented by a gate `cg` into an `MPO` form.
    """
    function MPO(cg::AbstractGate)
        MPO(Qaintessent.matrix(cg))
    end

    """
        MPO(cg::CircuitGate)

    Transform an operator represented by a gate `cg` into an `MPO` form.
    """
    function MPO(cg::AbstractCircuitGate)
        MPO(Qaintessent.matrix(cg))
    end
end

"""
    extend_MPO(mpo::MPO, iwire::NTuple{M, <:Integer}) where M

Extend an operator `MPO` acting on `M` qudits into an operator acting on `N` qudits by inserting identities.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `MPO` acts. It must be sorted.
...

"""

function extend_MPO(mpo::MPO, iwire::NTuple{M, <:Integer}) where M

    length(unique(iwire)) == length(iwire) || error("Repeated wires are not valid.")
    collect(iwire) == sort(collect(iwire), rev=true) || error("Wires not sorted")
    prod(0 .< iwire) || error("Wires must be positive integers.")

    iwire = reverse(iwire)

    N = length(iwire[1]:iwire[end])
    @assert length(mpo.tensors) == M

    M != N || error("MPO is already decomposed in N tensors")
    M ≥ 1 || error("MPO needs at least one qubit to act on.")

    # TODO: support general "qudits"
    d = 2
    qwire = Tuple(iwire[1]:iwire[end])
    pipeswire = setdiff(qwire, iwire)
    pipeswire = sort(pipeswire)
    qwire = reverse(qwire)

    for (i,w) in enumerate(pipeswire)
        ind = findfirst(x->x==w, qwire)
        bond = size(mpo.tensors[ind-1])[end]
        Vpipe = reshape(kron(Matrix(1I, bond, bond), Matrix(1I, d, d)), (bond, d, bond, d))
        Vpipe = permutedims(Vpipe, [1,2,4,3])
        insert!(mpo.tensors, ind, Tensor(Vpipe))
    end

    for i in M:N-1
        push!(mpo.contractions, Summation([i => 4, i+1 => 1]))
        pushfirst!(mpo.openidx, i + 1 => 2)
        insert!(mpo.openidx, i+2, i + 1 => 3)
    end
    return mpo
end

"""@docs
    extend_MPO(m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M

Extend an operator represented by a matrix `m` acting on `M` qudits into an operator acting on `N` qudits by inserting identities.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `m` acts. It must be sorted.
...
"""

function extend_MPO(m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M
    length(unique(iwire)) == length(iwire) || error("Repeated wires are not valid.")
    prod(0 .< iwire) || error("Wires must be positive integers.")
    collect(iwire) == sort(collect(iwire), rev=true) || error("Wires not sorted")
return extend_MPO(MPO(m), iwire)
end


"""@docs
    apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M

Given a state `ψ`  in a Tensor Network form and an operator `mpo` acting on `M` qudits, update the state by effectively applying `mpo`.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `MPO` acts. When the input is `mpo::MPO`, `iwires` must be sorted.
...
"""

function apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M

    length(unique(iwire)) == length(iwire) || error("Repeated wires are not valid.")

    # sort([iwire...]) == [iwire...] || error("Input 'iwires must be sorted'")

    n = length(ψ.openidx) #number of qudits

    prod(0 .< iwire .<= n) || error("Wires must be integers between 1 and n (total number of qudits).")

    step = length(ψ.tensors)
    # qwire = (iwire[1]:iwire[end]...,)
    N = length(iwire)
    # TODO: support general "qudits"
    d = 2
    iwire = reverse(iwire)
    if M < N
        mpo = extend_MPO(mpo, Tuple(iwire))
    end

    ψ_prime = GeneralTensorNetwork([copy(ψ.tensors); copy(mpo.tensors)],
                        [copy(ψ.contractions); shift_summation.(mpo.contractions, step)],
                        copy(ψ.openidx))
    #add the contractions between mpo and state
    for (i,w) in enumerate(iwire)
        push!(ψ_prime.contractions, Summation([ψ.openidx[w], shift_pair(mpo.openidx[i+N], step)]))
    end
    #update the openidx
    for (i,q) in enumerate(iwire)
        ψ_prime.openidx[q] = shift_pair(mpo.openidx[i], step)
    end
    return ψ_prime
end

"""
    apply_MPO(ψ::TensorNetwork, m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M

Given a state `ψ`  in a Tensor Network form and an operator represented by a matrix `m` acting on `M` qudits, update the state by effectively applying `m`.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `MPO` acts. It does not need to be sorted; in this case, the function performs the corresponding permutation
of the dimensions.
...
"""
function apply_MPO(ψ::TensorNetwork, m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M
    length(unique(iwire)) == length(iwire) || error("Repeated wires are not valid.")
    prod(0 .< iwire) || error("Wires must be positive integers.")

    iwire_sorted = sort(collect(iwire))
    if iwire_sorted != collect(iwire)
        sort_wires = sortperm(collect(iwire))
        perm = [sort_wires...; (sort_wires.+M)]
        m = reshape(m, fill(2, 2M)...)
        m = permutedims(m, perm)
        m = reshape(m, (2^M, 2^M))
    end
    return apply_MPO(ψ, MPO(m), Tuple(iwire_sorted))
end

"""
    apply_MPO(ψ::TensorNetwork, cg::CircuitGate)

Given a state `ψ`  in a Tensor Network form and CircuitGate `cg`, update the state by effectively applying `cg`.
"""
function apply_MPO(ψ::TensorNetwork, cg::CircuitGate)
    m = (cg.gate).matrix
    iwire = cg.iwire
return apply_MPO(ψ, m, iwire)
end

function Base.isapprox(mpo1::MPO, mpo2::MPO)
    all(mpo1.tensors .≈ mpo2.tensors) && all(all(mpo1.contractions .== mpo2.contractions)) && all(mpo1.openidx == mpo2.openidx)
end
