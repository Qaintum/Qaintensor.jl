function MPS(T::AbstractVector{Tensor})
    l=length(T)
    for i in 1:l
        @assert ndims(T[i])==3
    end

    contractions=[Summation([i=>3,i+1=>1]) for i in 1:l-1]
    openidx=reverse([1=>1; [i=>2 for i in 1:l]; l=>3])
    tn=TensorNetwork(T, contractions, openidx)
    return tn
end

function MPS(T::Tensor, l)
    #translational invariant MPS
    return MPS(fill(T,l))
end

"""
    contract_svd_mps(tn::TensorNetwork, er)

tn: TensorNetwork. Must be provided in a MPS form, that is, tn.tensors have three legs,
    tn.contractions are of the form[(T_i, 3), (T_i+1,1)]
er: maximum error in the truncation done in an individual contraction
"""
function contract_svd_mps(tn::TensorNetwork, er)

    for (i,id) in enumerate(tn.contractions)
        @assert length(id.idx)==2
        @assert (id.idx[1].second==3) & (id.idx[2].second==1)
        #error if the contractions are not in the right order
    end

    lchain=length(tn.tensors)
    tcontract=tn.tensors[1]
    for j in 2:lchain
        tcontract=contract_svd(tcontract, tn.tensors[j], (1+j,1), er)
    end
    return tcontract
end
