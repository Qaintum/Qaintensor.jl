using LinearAlgebra

"""
    MPO(tensors::AbstractVector{Tensor}, contractions::AbstractVector{Summation}, openidx::AbstractVector{Pair{T,T}}) where T <: Integer

Subclassed TensorNetwork object in Matrix Product Operator(MPO) form.
"""
struct MPO <: TensorNetwork
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
                #push!(openidx, i => 2)
            end
            push!(openidx, 1 => 1)

            #input dimensions

            for i in 1:M-1
                push!(openidx, M-i+1 => 3)
                #push!(openidx, i => 3)
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

    function MPO(cg::CircuitGate)
        MPO(Qaintessent.matrix(cg))
    end
end

function shift_summation(S::Summation, step)
   return Summation([S.idx[i].first + step => S.idx[i].second for i in 1:2])
end

function shift_pair(P::Pair, step)
    return P.first + step => P.second
end
"""
    circuit_MPO(mpo::MPO, iwire::NTuple{M, <:Integer}) where M

extends an operator `MPO` acting on `M` qudits into an operator acting on `N` qudits by inserting identities
"""

function circuit_MPO(mpo::MPO, iwire::NTuple{M, <:Integer}) where M
    collect(iwire) == sort(collect(iwire)) || @error("Wires not sorted")
    N = length(iwire[1]:iwire[end])
    @assert length(mpo.tensors) == M

    M != N || error("MPO is already decomposed in N tensors")
    M ≥ 1 || error("MPO needs at least one qubit to act on.")

    # TODO: support general "qudits"
    d = 2
    qwire = (iwire[1]:iwire[end]...,)
    pipeswire = setdiff(qwire, iwire)
    pipeswire = sort(pipeswire)
    qwire = reverse(qwire)

    for (i,w) in enumerate(reverse(pipeswire))
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

function circuit_MPO(m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M
    collect(iwire) == sort(collect(iwire)) || @error("Wires not sorted")
return circuit_MPO(MPO(m), iwire)
end

"""
    apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M

given a state `ψ`  in a Tensor Network form and an operator `mpo` acting on `M` qudits, it updates the state by effectively applying `mpo`. If `M` is smaller than the number of qudits of `psi`,  `circuit_MPO` is first applied.
"""

function apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M
    n = length(ψ.openidx) #number of qudits
    step = length(ψ.tensors)
    qwire = (iwire[1]:iwire[end]...,)
    N = length(qwire)
    # TODO: support general "qudits"
    d = 2

    if M < N
        mpo = circuit_MPO(mpo, Tuple(iwire))
    end

    ψ_prime = GeneralTensorNetwork([copy(ψ.tensors); copy(mpo.tensors)],
                        [copy(ψ.contractions); shift_summation.(mpo.contractions, step)],
                        copy(ψ.openidx))
    #add the contractions between mpo and state
    for (i,w) in enumerate(qwire)
        push!(ψ_prime.contractions, Summation([ψ.openidx[w], shift_pair(mpo.openidx[i+N], step)]))
    end
    #update the openidx
    for (i,q) in enumerate(qwire)
        ψ_prime.openidx[q] = shift_pair(mpo.openidx[i], step)
    end
    return ψ_prime
end

function apply_MPO(ψ::TensorNetwork, m::AbstractMatrix, iwire::NTuple{M, <:Integer}) where M
    iwire_sorted = sort(collect(iwire))
    if iwire_sorted != collect(iwire)
        sort_wires = sortperm(collect(iwire))
        perm = reverse([1:M...])
        perm = perm[sort_wires]
        perm = reverse(perm)
        perm = [perm; perm.+M]

        m = reshape(m, fill(2, 2M)...)
        m = permutedims(m, perm)
        m = reshape(m, (2^M, 2^M))
    end
    return apply_MPO(ψ, MPO(m), Tuple(iwire_sorted))
end
