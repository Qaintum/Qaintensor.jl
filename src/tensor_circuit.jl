
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
function tensor_circuit!(ψ::MPS, cg::CircuitGate{M,N,G}; er=nothing, max_bond_dim=typemax(Int64), revert=true, openidx=nothing) where {M,N,G}

    # TODO: specialization for various G
    max_bond_dim >= 2 || error("Max bond dimension must be minimally 2")
    if isnothing(openidx)
        openidx = [N:-1:1...]
    end

    if isnothing(er)
        er = 1e-5
    end

    # number of wires must agree
    @assert N == length(ψ.openidx)

    # TODO: support general "qudits"
    d = 2

    cg_matrix = reshape(Qaintessent.matrix(cg.gate), fill(d, 2*M)...)

    if M == 1
        x = findfirst(x->x==cg.iwire[1], openidx)

        if x == 1
            indexlist = [[1, -2], [1, -1]]
        elseif x == length(openidx)
            indexlist = [[-1, 1], [1, -2]]
        else
            indexlist = [[-1, 1, -3], [1, -2]]
        end

        ψ.tensors[x] = Tensor(ncon([ψ.tensors[x].data, cg_matrix], indexlist))

        return
    end

    tensors = decompose(cg_matrix, M)
    iwire = reverse(cg.iwire)

    x = findfirst(x->x==iwire[1], openidx)
    next_x = findfirst(x->x==iwire[2], openidx)

    δ = next_x - x

    dim1 = size(ψ.tensors[x])
    dim2 = size(tensors[1])

    if x == 1
        indexlist = [[1, -2], [1, -1, -3]]
        shape = (2, 4*dim1[2])
    elseif x == length(openidx)
        indexlist = [[-1, 1], [1, -3, -2]]
        shape = (4*dim1[1], 2)
    elseif δ > 0
        indexlist = [[-1, 1, -3], [1, -2, -4]]
        shape = (dim1[1], 2, 4*dim1[3])
    else
        indexlist = [[-1, 1, -4], [1, -3, -2]]
        shape = (4*dim1[1], 2, dim1[3])
    end


    contracted_tensor = ncon([ψ.tensors[x].data, tensors[1]], indexlist)
    ψ.tensors[x] = Tensor(reshape(contracted_tensor, shape))

    for (i, w) in enumerate(cg.iwire[2:end-1])

        if abs(δ) > 1
            index1 = length(openidx) - next_x + 1
            index2 = length(openidx) - (x+sign(δ)) + 1
            switch!(ψ, index1, index2)
            tmp = openidx[next_x:sign(-δ):x+sign(δ)]
            openidx[next_x:sign(-δ):x + sign(δ)] .= circshift(tmp, -1)
            tmp = openidx[x + 2sign(δ):sign(δ):next_x]
            openidx[x + 2sign(δ):sign(δ):next_x] .= circshift(tmp, -1)
        end

        last_δ = δ
        x = x+sign(δ)
        next_x = findfirst(x->x==iwire[i+2], openidx)
        δ = next_x - x

        dim1 = size(ψ.tensors[x].data)
        dim2 = size(tensors[i+1])

        if x == 1
            indexlist = [[1, -2], [-3, 1, -1, -4]]
            shape = (2, dim1[2]*dim2[1]*dim2[4])
        elseif x == length(openidx)
            indexlist = [[-1, 1], [-2, 1, -4, -3]]
            shape = (dim1[1]*dim2[1]*dim2[4], 2)
        elseif last_δ > 0
            if δ > 0
                indexlist = [[-1, 1, -4], [-2, 1, -3, -5]]
                shape = (dim1[1]*dim2[1], dim2[3], dim1[3]*dim2[4])

            else
                indexlist = [[-1, 1, -5], [-2, 1, -4, -3]]
                shape = (dim1[1]*dim2[4]*dim2[1], dim2[3], dim1[3])
            end
        else
            if δ < 0
                indexlist = [[-1, 1, -4], [-5, 1, -3, -2]]
                shape = (dim1[1]*dim2[4], dim2[3], dim1[3]*dim2[1])
            else
                indexlist = [[-1, 1, -3], [-4, 1, -2, -5]]
                shape = (dim1[1], dim2[3], dim1[3]*dim2[4]*dim2[1])
            end
        end

        contracted_tensor = ncon([ψ.tensors[x].data, tensors[i+1]], indexlist)
        ψ.tensors[x] = Tensor(reshape(contracted_tensor, shape))

        for (i,w) in enumerate(ψ.tensors)
            shape = size(ψ.tensors[i])
            if shape[end] > max_bond_dim
                ψ.tensors[i], ψ.tensors[i+1] = truncate_svd(ψ.tensors[i], ψ.tensors[i+1], er)
            end
        end

    end

    if abs(δ) > 1
        index1 = length(openidx) - next_x + 1
        index2 = length(openidx) - (x+sign(δ)) + 1
        switch!(ψ, index1, index2)
        tmp = openidx[next_x:sign(-δ):x+sign(δ)]

        openidx[next_x:sign(-δ):x + sign(δ)] .= circshift(tmp, -1)
        tmp = openidx[x + 2sign(δ):sign(δ):next_x]
        openidx[x + 2sign(δ):sign(δ):next_x] .= circshift(tmp, -1)
    end

    x = x+sign(δ)

    dim1 = size(ψ.tensors[x].data)
    dim2 = size(tensors[end])

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

    contracted_tensor = ncon([ψ.tensors[x].data, tensors[end]], indexlist)
    ψ.tensors[x] = Tensor(reshape(contracted_tensor, shape))

    if revert
        revert_indices = Int[]
        for i in length(openidx):-1:1
            push!(revert_indices, length(openidx) - findfirst(x->x==i, openidx) + 1)
        end
        permute!(ψ, reverse(revert_indices))
    end

end


"""
    tensor_circuit!(ψ, cgc)

Incorporate a circuit gate chain into a quantum tensor network state,
effectively applying the circuit to the state.
"""
function tensor_circuit!(ψ::MPS, cgc::CircuitGateChain{N}; max_bond_dim=typemax(Int64)) where {N}
    openidx = [N:-1:1...]
    for moment in cgc
        for gate in moment
            tensor_circuit!(ψ, gate; revert=false, openidx=openidx, max_bond_dim=max_bond_dim)
        end
    end
    println(openidx)
    revert_indices = Int[]
    for i in length(openidx):-1:1
        push!(revert_indices, length(openidx) - findfirst(x->x==i, openidx) + 1)
    end
    permute!(ψ, reverse(revert_indices))
end
