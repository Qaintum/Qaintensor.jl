"""
    Qaintessent.apply(mpo::MPO, ψ::TensorNetwork, iwire::NTuple{M, <:Integer}) where M

Given a state `ψ`  in a Tensor Network form and an operator `mpo` acting on `M` qudits, update the state by effectively applying `mpo`.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `MPO` acts. When the input is `mpo::MPO`, `iwires` must be sorted.
...
"""

function Qaintessent.apply(mpo::MPO, ψ::TensorNetwork, iwire::NTuple{M, <:Integer}) where M

    length(unique(iwire)) == length(iwire) || error("Repeated wires are not valid.")

    n = length(ψ.openidx) #number of qudits

    prod(0 .< iwire .<= n) || error("Wires must be integers between 1 and n (total number of qudits).")

    step = length(ψ.tensors)
    # qwire = (iwire[1]:iwire[end]...,)
    N = length(iwire)
    # TODO: support general "qudits"
    d = 2

    if M < N
        mpo = extend_MPO(mpo, Tuple(iwire))
    end
    
    iwire = reverse(iwire)

    ψ_prime = GeneralTensorNetwork([copy(ψ.tensors); copy(mpo.tensors)],
                        [copy(ψ.contractions); shift_summation.(mpo.contractions, step)],
                        copy(ψ.openidx))
    #add the contractions between mpo and state
    for (i,w) in enumerate(iwire)
        push!(ψ_prime.contractions, Summation([ψ.openidx[w], shift_pair(mpo.openidx[i+N], step)]))
    end
    #update the openidx
    for (i,q) in enumerate(iwire)
        ψ_prime.openidx[q] = shift_pair(mpo.openidx[i], step)
    end
    return ψ_prime
end

"""
    Qaintessent.apply(m::AbstractMatrix, ψ::TensorNetwork, iwire::NTuple{M, <:Integer}) where M

Given a state `ψ`  in a Tensor Network form and an operator represented by a matrix `m` acting on `M` qudits, update the state by effectively applying `m`.
...
# Arguments
- `iwire::NTuple{M, <:Integer}`: qudits in which `MPO` acts. It does not need to be sorted; in this case, the function performs the corresponding permutation
of the dimensions.
...
"""
function Qaintessent.apply(m::AbstractMatrix, ψ::TensorNetwork, iwire::NTuple{M, <:Integer}) where M
    length(unique(iwire)) == length(iwire) || error("Repeated wires are not valid.")
    prod(0 .< iwire) || error("Wires must be positive integers.")
    iwire = collect(iwire)
    iwire_sorted = sort(iwire)
    if iwire_sorted != iwire
        sort_wires = sortperm(iwire)
        perm = [sort_wires...; (sort_wires.+M)]
        m = reshape(m, fill(2, 2M)...)
        m = permutedims(m, perm)
        m = reshape(m, (2^M, 2^M))
    end
    return Qaintessent.apply(MPO(m), ψ, Tuple(iwire_sorted))
end

"""
    Qaintessent.apply(cg::CircuitGateψ, ::TensorNetwork)

Given a state `ψ`  in a Tensor Network form and CircuitGate `cg`, update the state by effectively applying `cg`.
"""
function Qaintessent.apply(cg::CircuitGate, ψ::TensorNetwork)
    m = (cg.gate).matrix
    iwire = cg.iwire
    return Qaintessent.apply(m, ψ, iwire)
end
