"""  MPO(m::AbstractMatrix)
Given the matrix m of an operator, MPO transforms it into a Matrix Product Operator form. 
"""
function MPO(m::AbstractMatrix)

    # TODO: support general "qudits"
    d = 2
    @assert size(A)[1] == size(A)[2]

    M = Int(log2(size(A)[1]))
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

        m = diagm(S) * transpose(V)
        push!(t, Tensor(reshape(U, (2, 2, bond_dim))))
        push!(con, Summation([1 => 3, 2 => 1]))
        push!(openidx, 1 => 1)
        push!(openidx, 1 => 2)

        for i in 2:M-1
            m = reshape(m, (bond_dim * 4, :))
            U, S, V = svd(Array(m))
            m = diagm(S) * transpose(V)
            new_bond_dim = size(S)[1]

            push!(t, Tensor(reshape(U, (bond_dim, 2, 2, new_bond_dim))))
            push!(con, Summation([i => 4, i+1 => 1]))
            push!(openidx, i => 2)
            push!(openidx, i => 3)

            bond_dim = new_bond_dim
        end

        push!(t, Tensor(reshape(m, (bond_dim, 2, 2))))
        push!(openidx, M => 2)
        push!(openidx, M => 3)

        else
        #one-qubit operator
        m = reshape(m, 2, 2)
        push!(t, Tensor(m))
        openidx = [1 => 1, 1 => 2]
    end

    return TensorNetwork(t, con, openidx)
end
