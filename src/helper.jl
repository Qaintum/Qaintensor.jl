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
    is_power_two(i::Integer)

Return true if i is a power of 2, else false.
"""
function is_power_two(i::Integer)
    i != 0 || return false
    return (i & (i - 1)) == 0
end
