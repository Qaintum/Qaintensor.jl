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

"""  circuit_MPO(MPO::TensorNetwork, N, iwire::NTuple{M, <:Integer}) where M
Given an MPO acting on M qubits, not necessesarily adjacent, it converts it in an MPO acting on N qubits.
"""
function circuit_MPO(mpo::MPO, N, iwire::NTuple{M, <:Integer}) where M

    @assert length(mpo.tensors) == M
    M != N || error("MPO is already decomposed in N tensors")
    M ≥ 1 || error("MPO needs at least one qubit to act on.")
    M ≤ N || error("Number of qubits in which the MPO acts cannot be larger than total number of qubits.")

    d = 2

    if iwire[1] > 1
        mpo.tensors[1] = reshape(mpo.tensor[1], (1, size(mpo.tensor[1])...))
        Vpipe = reshape(Matrix(1I, d, d), (d, d, 1))
        pushfirst!(mpo.tensors, Vpipe)
    end

    if iwire[end] < N
        mpo.tensor[end] = reshape(mpo.tensor[end], (size(MPO.tensor[end])..., 1))
        Vpipe = reshape(Matrix(1I, d, d), (1, d, d))
        push!(mpo.tensors, Vpipe)
    end

    for w in setdiff((1:N...,), iwire)
        if (w != 1) & (w != N)
            bond = size(mpo.tensors[w])[1]
            Vpipe = reshape(kron(Matrix(1I, bond, bond), Matrix(1I, d, d)), (bond, d, bond, d))
            Vpipe = permutedims(Vpipe, [1,2,4,3])
            insert!(mpo.tensors, w, Tensor(Vpipe))
        end
    end

    for i in M:N-1
        push!(mpo.contractions, Summation([i => 4, i+1 => 1]))
        pushfirst!(mpo.openidx, i + 1 => 3)
        insert!(mpo.openidx, M + i, i + 1 => 2)
    end
    return MPO
end

"""  apply_MPO(ψ::TensorNetwork, MPO::TensorNetwork, iwire::NTuple{M, <:Integer}) where M
Given a state ψ of N qudits in a Tensor Network form and an operator MPO acting on M qudits,
it updates the state by effectively applying the MPO. If M < N the MPO is first converted into
a circuit MPO acting on N qudits.  """
function apply_MPO(ψ::TensorNetwork, mpo::MPO, iwire::NTuple{M, <:Integer}) where M

    # TODO: support general "qudits"
    d = 2

    N = length(ψ.openidx)
    step = length(ψ.tensors)

    if M < N
        MPO = circuit_MPO(mpo, N, iwire)
    end

    ψ_prime = GeneralTensorNetwork(copy(ψ.tensors),[copy(ψ.contractions); shift_summation.(mpo.contractions, step)], [])
    for i in 1:N-1
        push!(ψ_prime.tensors, mpo.tensors[i])
        push!(ψ_prime.contractions, Summation([ψ.openidx[i], N + step-i+1 => 3]))
        push!(ψ_prime.openidx, N + step - i+1 => 2)
    end
    push!(ψ_prime.tensors, mpo.tensors[N])
    push!(ψ_prime.openidx, 1+step => 1)
    push!(ψ_prime.contractions, Summation([ψ.openidx[N], step + 1 => 2]))

    return ψ_prime
end
