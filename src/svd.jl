using LinearAlgebra: svd
using TSVD
"""
    contract_svd(T1::Tensor, T2::Tensor, indx::NTuple{2,Int}, er)

Contract tensors `T1` and `T2` along indices in `idx` with error `er`
# Arguments
- T1, T2:  tensors to contract
- idx: (id1, id2) indices of T1 and T2 to contract
- er: maximum error in the truncation
"""
function contract_svd(T1::Tensor, T2::Tensor, indx::NTuple{2,Int}; er=0.0)
    i1,i2 = indx
    n1, n2 = Qaintensor.ndims(T1), Qaintensor.ndims(T2)
    D1, D2 = size(T1.data, i1), size(T2.data,i2)
    size1, size2 = size(T1.data), size(T2.data)
    newdim = [size1[1:i1-1]...; size1[i1+1:n1]...;size2[1:i2-1]...; size2[i2+1:n2]...]
    D1 == D2 || error("Dimensions of contraction legs do not match")

    #TODO: if shape of tensors is already in appropiate form skip this

    T1p = permutedims(T1.data, [1:i1-1...,i1+1:n1...,i1])
    T2p = permutedims(T2.data, [i2, 1:i2-1...,i2+1:n2...])

    T1p = reshape(T1p,(:,D1))
    T2p = reshape(T2p,(D1,:))

    SVD1, SVD2 = svd(T1p), svd(T2p)
    U1, S1, V1 = SVD1.U, SVD1.S, SVD1.Vt
    U2, S2, V2 = SVD2.U, SVD2.S, SVD2.Vt

    cumsum1, cumsum2 = cumsum(reverse(S1)), cumsum(reverse(S2))
    k1_reverse = findfirst(x-> x>er, cumsum1)
    k1 = length(cumsum1)-k1_reverse+1
    k2_reverse = findfirst(x-> x>er, cumsum2)
    k2 = length(cumsum2)-k2_reverse+1

    T = U1[:,1:k1]*(V1[1:k1,:].*S1[1:k1])*(S2[1:k2]'.*U2[:,1:k2])*V2[1:k2,:]
    T = reshape(T, newdim...)
    return Tensor(T)
end

"""
    truncate_svd(T1::Tensor, T2::Tensor)

Contract tensors `T1` and `T2`. Assumes contraction on the last index of T1 and first index of T2
# Arguments
- T1, T2:  tensors to contract
- idx: (id1, id2) indices of T1 and T2 to contract
- er: maximum error in the truncation
"""
function truncate_svd(T1::Tensor, T2::Tensor, max_bond_dim)

    n1, n2 = Qaintensor.ndims(T1), Qaintensor.ndims(T2)
    i1,i2 = n1, 1
    D1, D2 = size(T1.data, i1), size(T2.data, i2)
    size1, size2 = size(T1.data), size(T2.data)
    if D1 != D2
        return T1, T2
    end
    print("Truncating size: " * string(size(T1)) * " and " * string(size(T2)))
    #TODO: if shape of tensors is already in appropiate form skip this
    tensors = [fill(0, ndims(t)) for t in [T1,T2]]
    tensors[1][i1] = 1
    tensors[2][i2] = 1
    counter = -1
    for t in 1:2
        for ind in 1:length(tensors[t])
            if tensors[t][ind] == 0
                tensors[t][ind] = counter
                counter -= 1
            end
        end
    end

    T = ncon([T1.data, T2.data], tensors)
    T = reshape(T, (prod(size1[1:end-1]), prod(size2[2:end]) ) )
    U, S, V = svd(T)
    max_bond_dim = minimum([max_bond_dim, length(S)])
    V = diagm(S[1:max_bond_dim]) * adjoint(V[:,1:max_bond_dim])

    println(norm(T - U[:, 1:max_bond_dim]*V))

    U = reshape(U[:,1:max_bond_dim], (size1[1:end-1]..., max_bond_dim))
    V = reshape(V, (max_bond_dim, size2[2:end]...))

    return Tensor(U), Tensor(V)
end
