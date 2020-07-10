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
            for i in 1:M-1
                push!(openidx, M-i+1 => 3)
            end
            push!(openidx, 1 => 2)
            for i in 1:M-1
                push!(openidx, M-i+1 => 2)
            end
            push!(openidx, 1 => 1)

            push!(t, Tensor(reshape(m, (bond_dim, 2, 2))))

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

"""
    circuit_MPO(mpo::MPO, iwire::NTuple{M, <:Integer}) where M

extends an operator `MPO` acting on `M` qudits into an operator acting on `N` qudits by inserting identities
"""

function circuit_MPO(mpo::MPO, iwire::NTuple{M, <:Integer}) where M
    N = length(iwire[1]:iwire[end])    
    @assert length(mpo.tensors) == M

    M != N || error("MPO is already decomposed in N tensors")
    M ≥ 1 || error("MPO needs at least one qubit to act on.")

    # TODO: support general "qudits"
    d = 2

    pipeswire = setdiff((iwire[1]:iwire[end]...,), iwire)
    for (i,w) in enumerate(pipeswire)
        bond = size(mpo.tensors[w-iwire[1]])[end]
        Vpipe = reshape(kron(Matrix(1I, bond, bond), Matrix(1I, d, d)), (bond, d, bond, d))
        #Julia ordena las dimensiones de salida->entrada
        Vpipe = permutedims(Vpipe, [1,2,4,3])
        insert!(mpo.tensors, w-iwire[1]+1, Tensor(Vpipe))
    end

    for i in M:N-1
        push!(mpo.contractions, Summation([i => 4, i+1 => 1]))
        pushfirst!(mpo.openidx, i + 1 => 3)
        insert!(mpo.openidx, M + i, i + 1 => 2)
    end

    return MPO
end

"""
    apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M

given a state `ψ`  in a Tensor Network form and an operator `mpo` acting on `M` qudits, it updates the state by effectively applying `mpo`. If `M` is smaller than the number of qudits of `psi`,  `circuit_MPO` is first applied.
"""

function apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M

    iwire = sort(collect(iwire))
    # TODO: support general "qudits"
    d = 2
    n = length(ψ.openidx) #number of qudits
    step = length(ψ.tensors)

    N = length(iwire[1]:iwire[end])
    if M < N
        mpo = circuit_MPO(mpo, Tuple(iwire))
    end

    qwire = (iwire[1]:iwire[end]...,)
    qbef = (1:iwire[1]-1...,)
    qaft = (iwire[end]+1:n...,)

    ψ_prime = TensorNetwork([copy(ψ.tensors); copy(mpo.tensors)],
                        [copy(ψ.contractions); shift_summation.(mpo.contractions, step)],
                        [])
    for (i,w) in enumerate(qwire[2:end])
        push!(ψ_prime.contractions, Summation([ψ.openidx[n-w+1], step+i+1 => 3]))
    end
    push!(ψ_prime.contractions, Summation([ψ.openidx[n-qwire[1]+1], step + 1 => 2]))

    for (i,q) in enumerate(qaft)
        push!(ψ_prime.openidx, ψ.openidx[n-q+1])
    end

    for (i,w) in enumerate(qwire[2:end])
            push!(ψ_prime.openidx,  N + step - i + 1 => 2)
    end
    push!(ψ_prime.openidx, step + 1 => 1)


    for (i,q) in enumerate(qbef)
        push!(ψ_prime.openidx, ψ.openidx[n-q+1])
    end

    return ψ_prime
end
