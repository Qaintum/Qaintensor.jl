function extend_MPO_binarytree(MPO::MPO, lpath)
    bond = size(MPO.tensors[1])[3]
    Vpipe = reshape(kron(Matrix(1I, bond, bond), Matrix(1I, 1, 1)), (bond, 1, bond, 1))
    Vpipe = permutedims(Vpipe, [1,2,4,3])

    for i in 1:lpath
        insert!(MPO.tensors, 2, Tensor(Vpipe))
    end

    for i in 2:(lpath+1)
        push!(MPO.contractions, Summation([i => 4, i+1 => 1]))
        pushfirst!(MPO.openidx, i + 1 => 2)
        insert!(MPO.openidx, i+2, i + 1 => 3)
    end
    return MPO
end

"""
    apply_MPO_binarytree(ψ::GeneralTensorNetwork, MPO::MPO, iwire::NTuple{M, <:Integer}) where M

Apply an operator decomposed as an MPO to  `ψ`, by effectively updating the tensor network. """

function apply_MPO_binarytree(ψ::GeneralTensorNetwork, MPO::MPO, iwire::NTuple{M, <:Integer}) where M
    # TODO: check that input ψ is in binary tree form

    # TODO: implement general M-qubit operators
    length(iwire) == 2 || @error("This function only supports 2-qubit operators for the moment.")

    # TODO: adapt for not-sorted iwires
    collect(iwire) == sort(collect(iwire)) || @error("This function only supports sorted iwires for the moment.")

    step = length(ψ.tensors)
    n_tensors = length(ψ.openidx) #number of qubits
    w = log2(n_tensors+1) #number of layers

    path = function_path(iwire[1], iwire[2], w)
    lpath = length(path)

    mpo = extend_MPO_binarytree(MPO, length(path)-2)
    ψ_prime = GeneralTensorNetwork([copy(ψ.tensors); copy(mpo.tensors)], copy(ψ.contractions), copy(ψ.openidx))

    for (i, t) in enumerate(path)
        push!(ψ_prime.contractions, Summation([ψ.openidx[n_tensors+1-t], Qaintensor.shift_pair(mpo.openidx[lpath+i], step)]))
    end

    for (i, t) in enumerate(path)
        ψ_prime.openidx[n_tensors+1-t] = Qaintensor.shift_pair(mpo.openidx[i], step)
    end

    ψ_prime.contractions = [ψ_prime.contractions; shift_summation.(MPO.contractions, step)]

    return ψ_prime

end
