using LinearAlgebra
"""
 decompose: decomposes multiple qubit gates into smaller tensors
"""

function decompose!(cg::CircuitGate{M,N,G}) where {M,N,G}
    M > 1 || error("Only decompose Circuit Gates that apply to multiple wires")
    m = Array(Qaintessent.matrix(cg.gate))

    d = 2
    bond_dim = 1
    t = Tensor[]
    w = Integer[]
    c = Integer[]

    m = reshape(m, fill(2, 2M)...)

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
    pushfirst!(w, cg.iwire[1])
    push!(c, 0)
    push!(c, 3)

    for i in 2:M-1
        m = reshape(m, (bond_dim * 4, :))
        U, S, V = svd(Array(m))
        m = diagm(S) * adjoint(V)
        new_bond_dim = size(S)[1]

        push!(t, Tensor(reshape(U, (bond_dim, 2, 2, new_bond_dim))))
        pushfirst!(w, cg.iwire[i])
        push!(c, 4)
        bond_dim = new_bond_dim
    end

    push!(t, Tensor(reshape(m, (bond_dim, 2, 2))))
    pushfirst!(w, cg.iwire[M])

    return (t,c,w)
end
