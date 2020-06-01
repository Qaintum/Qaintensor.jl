function shift_summation(S::Summation, step::Integer)
   return Summation([S.idx[i].first + step => S.idx[i].second for i in 1:2])
end

function shift_pair(P::Pair{Integer, Integer}, step::Integer)
    return P.first + step => P.second
end

function exp_value(ψ::TensorNetwork, O::CircuitGate{M,N,G}) where {M,N,G}

    #TODO: support one-qubit operators

    ψ_prime = TensorNetwork(copy(ψ.tensors), copy(ψ.contractions), copy(ψ.openidx) )
    tensor_circuit!(ψ_prime, O)
    l = length(ψ.tensors)
    step = length(ψ_prime.tensors)
    for i in 1:l
        push!(ψ_prime.tensors, conj(ψ.tensors[i]))
    end

    for i in 1:length(ψ.openidx)
        push!(ψ_prime.contractions, Summation([ψ_prime.openidx[i], shift_pair(ψ.openidx[i], step)]))
    end

    println(ψ_prime.contractions)

    for (i, con) in enumerate(ψ.contractions)
        push!(ψ_prime.contractions, shift_summation(con, step))
    end

    ψ_prime.openidx=Pair[]

    return contract(ψ_prime)
end
