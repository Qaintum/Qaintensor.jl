"""
    shift_summation(S::Summation, step)

Shift the first element of both pairs in summation `S` by `step`
"""
function shift_summation(S::Summation, step)
   return Summation([S.idx[i].first + step => S.idx[i].second for i in 1:2])
end

"""
    shift_pair(P::Pair, step)

Shift the first element of pair `P`  by `step`.
"""
function shift_pair(P::Pair, step)
    return P.first + step => P.second
end

"""
   nqubits(ψ::AbstractVector)

Given a vector `ψ` that represents a quantum state,
return the number of qubits `N` of the state.
"""

function nqubits(ψ::AbstractVector)
    N = Int(log2(length(ψ)))
    return N
end
