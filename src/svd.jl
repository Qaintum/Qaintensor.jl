using LinearAlgebra: svd
"""
    contract_svd(T1::Tensor, T2::Tensor, indx::NTuple{2,Int}, er)

Contract tensors `T1` and `T2` along indices in `idx` with error `er`
# Arguments
- T1, T2:  tensors to contract
- idx: (id1, id2) indices of T1 and T2 to contract
- er: maximum error in the truncation
"""
function contract_svd(T1::Tensor, T2::Tensor, indx::NTuple{2,Int}, er)
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
