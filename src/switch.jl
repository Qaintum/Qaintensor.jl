"""
    switch!(mps::MPS, indices::AbstractVector{Tuple})

swaps wire positions `i` and `j` in MPS object. this implementation does swaps one by one
"""
function switch!(mps::MPS, indices::AbstractVector{Tuple{Int64, Int64}})
    check_mps(mps)
    for index in indices
        switch!(mps, index[1], index[2])
    end
end

"""
    switch(mps::MPS, i::Integer)

switches `i`ith qubit in MPS with `i+1`th qubit in MPS
"""
function switch!(mps::MPS, i::Integer)
    i > 0 || BoundsError("Wire indices must be positive")
    i < length(mps.tensors) || BoundsError("Attempt to access " * string(i+1) * " qubit in MPS of size " * string(i))
    T1 = mps.tensors[i]
    T2 = mps.tensors[i+1]
    T1_dims = size(T1)
    T2_dims = size(T2)

    T = contract_svd(T1, T2, (ndims(T1), 1)).data

    if ndims(T1) == 2
        T = permutedims(T, [2,1,3])
        T = reshape(T, (2, 2*T2_dims[end]))
    elseif ndims(T2) == 2
        T = permutedims(T, [1,3,2])
        T = reshape(T, (2*T1_dims[1], 2))
    else
        T = permutedims(T, [1,3,2,4])
        T = reshape(T, (2*T1_dims[1], 2*T2_dims[end]))
    end

    U, S, V = svd(T)
    bond_dim = length(S)

    if ndims(T1) == 2
        U = reshape(U, (2, bond_dim))
        V = diagm(S) * adjoint(V)
        V = reshape(V, (bond_dim, 2, T2_dims[end]))
    elseif ndims(T2) == 2
        U = reshape(U, (T1_dims[1], 2, bond_dim))
        V = diagm(S) * adjoint(V)
    else
        U = reshape(U, (T1_dims[1], 2, bond_dim))
        V = diagm(S) * adjoint(V)
        V = reshape(V, (bond_dim, 2, T2_dims[end]))
    end
    mps.tensors[i] = Tensor(U)
    mps.tensors[i+1] = Tensor(V)
end

"""
    switch(mps::MPS, i::Integer, j::Integer)

swaps wire positions `i` and `j` in MPS object. this implementation does swaps one by one
"""
function switch!(mps::MPS, i::Integer, j::Integer)
    check_mps(mps)
    println("Swapping " * string(i) * " and " * string(j))
    i > 0 && j > 0 || error("Wire indices `i` and `j` must be positive")
    (i <= length(mps.tensors)) & (j <= length(mps.tensors)) || error("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS")

    i == j && return mps
    i = length(mps.tensors) - i + 1
    j = length(mps.tensors) - j + 1
    if i < j
        lo = i
        hi = j
    else
        lo = j
        hi = i
    end

    for i in lo:hi-1
        switch!(mps, i)
    end

    for j in hi-2:-1:lo
        switch!(mps, j)
    end
end


"""
    Base.permute!(mps::MPS, order::AbstractVector{Integer})

orders MPS qubits per order given in order <: AbstractVector
"""
function Base.permute!(mps::MPS, order::AbstractVector{Int64})
    check_mps(mps)
    length(order) == length(mps.tensors) || error("Given permutation must be same length as number of Tensors in MPS")
    length(unique(order)) == length(order) || error("Permutation order cannot contain repeat values")
    all(x->x>0, order) || error("Permutation order can only contain positive values")
    maximum(order) == length(order) || error("Wire numbers in permutation order cannot exceed number of wires in MPS")
    seq = Array(1:length(order))
    for i in 1:length(order)
        loc = findall(x->x==order[i], seq)[1]
        switch!(mps, i, loc)
        seq[loc], seq[i] = seq[i], seq[loc]
    end
end
