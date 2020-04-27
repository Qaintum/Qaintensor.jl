
"""
    tensor_circuit!(ψ, cg)

Update the tensor network description of a quantum state by a circuit gate,
effectively applying the gate.
"""
function tensor_circuit!(ψ::TensorNetwork, cg::CircuitGate{M,N,G}) where {M,N,G}
    # TODO: specialization for various G

    # number of wires must agree
    @assert N == length(ψ.openidx)

    # TODO: support general "qudits"
    d = 2

    # add gate as tensor to network
    push!(ψ.tensors, Tensor(reshape(Qaintessent.matrix(cg.gate), fill(d, 2*M)...)))

    for (i, w) in enumerate(cg.iwire)
        # contractions with new tensor;
        # last qubit corresponds to fastest varying index
        push!(ψ.contractions, Summation([ψ.openidx[w], length(ψ.tensors) => 2*M - i + 1]))
        # new open leg
        ψ.openidx[w] = length(ψ.tensors) => M - i + 1
    end
end


"""
    tensor_circuit!(ψ, cgc)

Incorporate a circuit gate chain into a quantum tensor network state,
effectively applying the circuit to the state.
"""
function tensor_circuit!(ψ::TensorNetwork, cgc::CircuitGateChain{N}) where {N}
    for gate in cgc
        tensor_circuit!(ψ, gate)
    end
end
