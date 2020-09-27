using LinearAlgebra: svd
"""
    contract_svd(T1::Tensor, T2::Tensor, indx::NTuple{2,Int}, er)

Contract tensors `T1` and `T2` along indices in `idx` with error `er`.
"""
function contract_svd(T1::Tensor, T2::Tensor, indx::NTuple{2,Int}; er = 0.0)

    er >= 0 || error("Error must be positive")

    i1,i2 = indx
    n1, n2 = ndims(T1), ndims(T2)
    D1, D2 = size(T1.data, i1), size(T2.data,i2)
    size1, size2 = size(T1.data), size(T2.data)
    newdim = [size1[1:i1-1]...; size1[i1+1:n1]...;size2[1:i2-1]...; size2[i2+1:n2]...]
    D1 == D2 || error("Dimensions of contraction legs do not match")

    #TODO: if shape of tensors is already in appropiate form skip this

    T1p = permutedims(T1.data, [1:i1-1...,i1+1:n1...,i1])
    T2p = permutedims(T2.data, [i2, 1:i2-1...,i2+1:n2...])

    T1p = reshape(T1p,(:,D1))
    T2p = reshape(T2p,(D1,:))

    U1, S1, V1 = svd(T1p)
    U2, S2, V2 = svd(T2p)

    cumsum1, cumsum2 = sqrt.(cumsum(reverse(S1.^2))), sqrt.(cumsum(reverse(S2.^2)))
    k1_reverse = findfirst(x-> x>er, cumsum1)
    k1 = length(cumsum1)-k1_reverse+1
    k2_reverse = findfirst(x-> x>er, cumsum2)
    k2 = length(cumsum2)-k2_reverse+1

    T = U1[:,1:k1]*diagm(S1)[1:k1, 1:k1]*adjoint(V1)[1:k1,:]*(U2[:,1:k2])*diagm(S2)[1:k2, 1:k2]*adjoint(V2)[1:k2,:]
    T = reshape(T, newdim...)
    return Tensor(T)
end
