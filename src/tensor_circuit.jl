
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
    for moment in cgc
        for gate in moment
            tensor_circuit!(ψ, gate)
        end
    end
end



"""
    tensor_circuit!(ψ, cg)

Update the tensor network description of a quantum state by a circuit gate,
effectively applying the gate.
"""
function tensor_circuit!(ψ::MPS, cg::CircuitGate{M,N,G}; revert=true, openidx=nothing) where {M,N,G}
    # TODO: specialization for various G
    if isnothing(openidx)
        openidx = [N:-1:1...]
    end

    # number of wires must agree
    @assert N == length(ψ.openidx)

    # TODO: support general "qudits"
    d = 2

    cg_matrix = reshape(Qaintessent.matrix(cg.gate), fill(d, 2*M)...)

    if M == 1
        x = findfirst(x->x==cg.iwire[1], openidx)

        # println(size(contract_svd(cg_matrix, ψ.tensors[tensor_index], (1, index))))
        if x == 1
            indexlist = [[1, -2], [1, -1]]
        elseif x == length(openidx)
            indexlist = [[-1, 1], [1, -2]]
        else
            indexlist = [[-1, 1, -3], [1, -2]]
        end

        # println(ncon([ψ.tensors[tensor_index].data, cg_matrix], indexlist))
        ψ.tensors[x] = Tensor(ncon([ψ.tensors[x].data, cg_matrix], indexlist))
        # new_tensor = contract_svd(ψ.tensors[tensor_index],cg_matrix, (index, 1))
        # ψ.tensors[tensor_index] = contract_svd(ψ.tensors[tensor_index],cg_matrix, (index, 1))
        return
    end

    tensors = decompose(cg_matrix, M)
    iwire = reverse(cg.iwire)

    # println(size.(tensors))
    x = findfirst(x->x==iwire[1], openidx)
    next_x = findfirst(x->x==iwire[2], openidx)

    δ = next_x - x

    dim1 = size(ψ.tensors[x])
    dim2 = size(tensors[1])
    println(δ)
    if x == 1
        indexlist = [[1, -2], [1, -1, -3]]
        shape = (2, 4*dim1[2])
    elseif x == length(openidx)
        indexlist = [[-1, 1], [1, -3, -2]]
        shape = (4*dim1[1], 2)
    elseif δ > 0
        indexlist = [[-1, 1, -3], [1, -2, -4]]
        shape = (dim1[1], 2, 4*dim1[3])
        # indexlist = [[-1, 1, -4], [1, -3, -2]]
        # shape = (4*dim1[1], 2, dim1[3])
    else
        indexlist = [[-1, 1, -4], [1, -3, -2]]
        shape = (4*dim1[1], 2, dim1[3])
        # indexlist = [[-1, 1, -3], [1, -2, -4]]
        # shape = (dim1[1], 2, 4*dim1[3])
    end

    # println(size.(ψ.tensors[x].data))
    # println(size.(tensors[1]))
    contracted_tensor = ncon([ψ.tensors[x].data, tensors[1]], indexlist)
    ψ.tensors[x] = Tensor(reshape(contracted_tensor, shape))

    for (i, w) in enumerate(cg.iwire[2:end-1])
        δ = next_x - x
        switch!(ψ, next_x, x+δ)
        tmp = openidx[next_x:sign(-δ):x+sign(δ)]
        openidx[next_x:sign(-δ):x + sign(δ)] .= circshift(tmp, -1)

        x = x+sign(δ)
        next_x = findfirst(x->x==cg.iwire[i+2], openidx)
        tensor_index = ψ.openidx[x][1]
        dim1 = size(ψ.tensors[tensor_index].data)
        dim2 = size(tensors[i+1])
        if δ > 0
            indexlist = [[-1, 1, -4], [-2, 1, -3, -5]]
            shape = (dim1[1]*dim2[1], dim2[3], dim1[3]*dim2[4])
        else
            indexlist = [[-1, 1, -4], [-5, 1, -3, -2]]
            shape = (dim1[1]*dim2[4], dim2[3], dim1[3]*dim2[1])
        end
        contracted_tensor = ncon([ψ.tensors[x].data, tensors[i+1]], indexlist)
        ψ.tensors[x] = Tensor(reshape(contracted_tensor, shape))
    end

    δ = next_x - x
    if abs(δ) > 1
        index1 = length(openidx) - next_x + 1
        index2 = length(openidx) - (x+sign(δ)) + 1
        switch!(ψ, index1, index2)
        tmp = openidx[next_x:sign(-δ):x+sign(δ)]

        openidx[next_x:sign(-δ):x + sign(δ)] .= circshift(tmp, -1)
        tmp = openidx[x + 2sign(δ):sign(δ):next_x]
        openidx[x + 2sign(δ):sign(δ):next_x] .= circshift(tmp, 1)
    end
    println(openidx)
    x = x+sign(δ)

    dim1 = size(ψ.tensors[x].data)
    dim2 = size(tensors[end])

    # if x == 1
    #     println("Here")
    #     indexlist = [[1, -3], [-2, 1, -1]]
    #     shape = (2, dim2[1]*dim1[2])
    # elseif x == length(openidx)
    #     indexlist = [[-3, 1], [-1, 1, -2]]
    #     shape = (dim2[1]*dim1[1], 2)
    # elseif δ > 0
    #     indexlist = [[-1, 1, -3], [-2, 1, -4]]
    #     shape = (dim2[1]*dim1[1], 2, dim1[3])
    # else
    #     indexlist = [[-1, 1, -4], [-3, 1, -2]]
    #     shape = (2*dim1[3], 2, dim2[3]*dim1[1])
    # end
    println(x)
    if x == 1
        indexlist = [[1, -2], [-3, 1, -1]]
        shape = (2, dim2[1]*dim1[2])
    elseif x == length(openidx)
        indexlist = [[-1, 1], [-2, 1, -3]]
        shape = (dim2[1]*dim1[1], 2)
    elseif δ > 0
        indexlist = [[-1, 1, -4], [-2, 1, -3]]
        shape = (dim2[1]*dim1[1], 2, dim1[3])
    else
        indexlist = [[-1, 1, -3], [-4, 1, -2]]
        shape = (dim1[1], 2, dim2[1]*dim1[3])
    end

    # println(indexlist)
    # println(shape)
    # println(x)
    # println(size(ψ.tensors[x].data))
    # println(size(tensors[end]))
    println(openidx)
    contracted_tensor = ncon([ψ.tensors[x].data, tensors[end]], indexlist)
    ψ.tensors[x] = Tensor(reshape(contracted_tensor, shape))
    revert_indices = Int[]
    for i in length(openidx):-1:1
        push!(revert_indices, length(openidx) - findfirst(x->x==i, openidx) + 1)
    end
    permute!(ψ, reverse(revert_indices))
end


"""
    tensor_circuit!(ψ, cgc)

Incorporate a circuit gate chain into a quantum tensor network state,
effectively applying the circuit to the state.
"""
function tensor_circuit!(ψ::MPS, cgc::CircuitGateChain{N}) where {N}
    openidx = [1:N...]
    for moment in cgc
        for gate in moment
            tensor_circuit!(ψ, gate; revert=false, openidx=openidx)
        end
    end
end
