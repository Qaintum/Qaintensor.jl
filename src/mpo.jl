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

    """
        MPO(m::AbstractMatrix)

    Transform an operator represented by matrix `m` into an `MPO` form."""
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
            push!(openidx, 1 => 1)
            for i in 2:M
                push!(openidx, i => 2)
            end
            # push!(openidx, M => 1)

            #input dimensions
            push!(openidx, 1 => 2)
            for i in 2:M
                push!(openidx, i => 3)
            end

        else
            #one-qubit operator
            m = reshape(m, 2, 2)
            push!(t, Tensor(m))
            openidx = [1 => 1, 1 => 2]
        end
        new(t, con, openidx)
    end

    """
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
    prod(0 .< iwire) || error("Wires must be positive integers.")

    if !issorted(iwire) && !issorted(iwire, rev=true)
        mpo, iwire = permute_MPO(mpo, iwire)
    end

    iwire = collect(iwire)

    rev = !issorted(iwire)
    wire_range = min(iwire...):max(iwire...)
    N = length(wire_range)

    @assert length(mpo.tensors) == M

    M != N || error("MPO is already decomposed in N tensors")
    M ≥ 1 || error("MPO needs at least one qubit to act on.")

    # TODO: support general "qudits"
    d = 2
    qwire = [wire_range...]
    pipeswire = setdiff(qwire, iwire)
    pipeswire = sort(pipeswire, rev=rev)
    qwire = sort(qwire, rev=rev)

    for (i,w) in enumerate(pipeswire)
        ind = findfirst(x->x==w, qwire)
        bond = size(mpo.tensors[ind-1])[end]
        Vpipe = reshape(kron(Matrix(1I, bond, bond), Matrix(1I, d, d)), (bond, d, bond, d))
        Vpipe = permutedims(Vpipe, [1,2,4,3])
        insert!(mpo.tensors, ind, Tensor(Vpipe))
    end

    for i in M:N-1
        push!(mpo.contractions, Summation([i => 4, i+1 => 1]))
        insert!(mpo.openidx, i + 1, i + 1 => 2)
        push!(mpo.openidx, i + 1 => 3)
    end

    if rev
        mpo.openidx = [reverse(mpo.openidx[1:N]);reverse(mpo.openidx[N+1:end])]
    end
    return mpo
end

"""
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
    if !issorted(iwire) && !issorted(iwire, rev=true)
        m, iwire = permute_matrix(m, iwire)
        extend_MPO(m, iwire)
    end
    return extend_MPO(MPO(m), iwire)
end

function Base.isapprox(mpo1::MPO, mpo2::MPO)
    all(mpo1.tensors .≈ mpo2.tensors) && all(all(mpo1.contractions .== mpo2.contractions)) && all(mpo1.openidx == mpo2.openidx)
end

"""
    permute_MPO(m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M

Extend an operator represented by a matrix `m` acting on `M` qudits into an operator acting on `N` qudits by inserting identities.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `m` acts. It must be sorted.
...
"""

function permute_MPO(m::MPO, iwire::NTuple{M, <:Integer}) where M
    m = contract(m)
    sort_wires = sortperm(collect(iwire))
    perm = [sort_wires...; (sort_wires.+M)]
    m = permutedims(m, perm)
    m = reshape(m, (2^M, 2^M))
    MPO(m), Tuple(sort(collect(iwire)))
end

"""
    permute_MPO(m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M

Extend an operator represented by a matrix `m` acting on `M` qudits into an operator acting on `N` qudits by inserting identities.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `m` acts. It must be sorted.
...
"""

function permute_matrix(m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M
    m = reshape(m, Tuple(fill(2, 2M)))
    sort_wires = sortperm(collect(iwire))
    perm = [sort_wires...; (sort_wires.+M)]
    m = permutedims(m, perm)
    reshape(m, (2^M, 2^M)), Tuple(sort(collect(iwire)))
end
