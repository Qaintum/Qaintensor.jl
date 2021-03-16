
"""
    tensor_circuit!(ψ, cg)

Update the tensor network description of a quantum state by a circuit gate,
effectively applying the gate.
```jldoctest; setup="using Random"
julia> N = 2
julia> ψ = MPS(rand(ComplexF64, 4)); 
julia> tensor_circuit!(ψ, circuit_gate(1, X))

```
"""
function tensor_circuit!(ψ::TensorNetwork, cg::CircuitGate{M,G}; is_decompose=false) where {M,G}
    # TODO: specialization for various G

    # number of wires must agree
    @assert Qaintessent.req_wires(cg) <= length(ψ.openidx)

    # TODO: support general "qudits"
    d = 2

    # add gate as tensor to network
    if M > 1 && is_decompose
        for (i, (t, c, w)) in enumerate(zip(decompose!(cg)...))
            push!(ψ.tensors, t)
            if i == 1
                push!(ψ.contractions, Summation([ψ.openidx[w], length(ψ.tensors) => 2]))
            else
                push!(ψ.contractions, Summation([ψ.openidx[w], length(ψ.tensors) => 3]))
            end
            if c > 0
                push!(ψ.contractions, Summation([length(ψ.tensors)-1 => c, length(ψ.tensors) => 1]))
            end
            if i == 1
                ψ.openidx[w] = length(ψ.tensors) => 1
            else
                ψ.openidx[w] = length(ψ.tensors) => 2
            end

        end
        return
    end
    push!(ψ.tensors, Tensor(reshape(Qaintessent.matrix(cg.gate), fill(d, 2*M)...)))
    for (i, w) in enumerate(cg.iwire)
        # contractions with new tensor;
        # last qubit corresponds to fastest varying index
        push!(ψ.contractions, Summation([ψ.openidx[w], length(ψ.tensors) => i]))
        # new open leg
        ψ.openidx[w] = length(ψ.tensors) => M + i
    end
end


"""
    tensor_circuit!(ψ, cgc)

Incorporate a circuit gate chain into a quantum tensor network state,
effectively applying the circuit to the state.
```jldoctest; setup="using Random"
julia> N = 2
julia> ψ = MPS(rand(ComplexF64, 4));
julia> cgs = qft_circuit(N) 
julia> tensor_circuit!(ψ, cgs)

```
"""
function tensor_circuit!(ψ::TensorNetwork, cgc::Vector{<:CircuitGate}; is_decompose=false)
    for gate in cgc
            tensor_circuit!(ψ, gate, is_decompose=is_decompose)
    end
end

function tensor_circuit!(ψ::TensorNetwork, cgc::Vector{Moment}; is_decompose=false)
    for moment in cgc
        for gate in moment
            tensor_circuit!(ψ, gate, is_decompose=is_decompose)
        end
    end
end