function add_layer!(tensors, tensors_def)
    is_inner_layer = (length(tensors_def) > 0)

    ten = pop!(tensors)
    s = size(ten)
    ten = reshape(ten, s[1], :)
    U, R = qr(ten)
    Vp = reshape(R, (size(U)[2], s[2:end]...))
    if is_inner_layer
        Vp = reshape(Vp, (size(Vp)...,1))
    end
    pushfirst!(tensors_def, Vp)
    pushfirst!(tensors, U)


    for i in 1:length(tensors)-2
        ten = pop!(tensors)
        s = size(ten)
        first = prod(s[1:2])
        second = prod(s[3:end])
        t = reshape(ten, (first, second))
        U, S, V = svd(t)
        Vp = reshape(diagm(S)*adjoint(V), (size(U)[2], s[3:end]...))
        if is_inner_layer
            Vp = reshape(Vp, (size(Vp)...,1))
        end
        Up = reshape(U, (s[1:2]..., size(U)[2]))
        pushfirst!(tensors_def, Vp)
        pushfirst!(tensors, Up)
    end

    ten = pop!(tensors)
    s = size(ten)
    ten = reshape(ten, s[1], :)
    U, R = qr(ten)
    Vp = reshape(R, (size(U)[2], s[2:end]...))
    if is_inner_layer
        Vp = reshape(Vp, (size(Vp)...,1))
    end
    pushfirst!(tensors_def, Vp)
    pushfirst!(tensors, U)

end

function contract_inner_layer!(tensors, tensors_def)
    N_pairs = length(tensors) ÷ 2

    if N_pairs ≥ 2
        U1 = popfirst!(tensors)
        U2 = popfirst!(tensors)
        s2 = size(U2)
        U2 = reshape(U2, (s2[1], s2[2]*s2[3]))
        U = U2'*U1
        U = reshape(U, (s2[2], s2[3], size(U1)[2]))
        U = permutedims(U, [1,3,2])
        push!(tensors, U);

        for i in 1:N_pairs-2
            U1 = popfirst!(tensors)
            U2 = popfirst!(tensors)
            s1 = size(U1)
            s2 = size(U2)

            U1 = permutedims(U1, [1,3,2])
            U1 = reshape(U1, (s1[1]*s1[3], s1[2]))

            U2 = reshape(U2, (s2[1], s2[2]*s2[3]))

            U = U1*U2
            U = reshape(U, (s1[1], s1[3], s2[2], s2[3]))
            U = permutedims(U, [1, 3, 2, 4])
            push!(tensors, U);
        end

        U1 = popfirst!(tensors)
        U2 = popfirst!(tensors)
        s1 = size(U1)
        s2 = size(U2)

        U1 = permutedims(U1, [1,3,2])
        U1 = reshape(U1, (s1[1]*s1[3], s1[2]))
        U = U1*U2
        U = reshape(U, (s1[1], s1[3], s2[2]))
        push!(tensors, U);

    else
        U1 = popfirst!(tensors)
        U2 = popfirst!(tensors)
        U = U1'*U2
        U = reshape(U, (size(U)...,1))
        pushfirst!(tensors_def, U);
    end
end

function contractions_binary_tree(w)
    cont = [Summation([1=>1, 2=>1]), Summation([1=>2, 3=>1])]
    openidx = []
    for k in 2:w-1
        for j in 1:2^(k-1)
            push!(cont, Summation([2^(k-1)+j-1 => 2 , 2^(k-1)+j-1+2^(k-1)-j + 2(j-1)+1 => 1]))
            push!(cont, Summation([2^(k-1)+j-1 => 3 , 2^(k-1)+j-1+2^(k-1)-j + 2(j-1)+2 => 1]))
        end
    end

    for i in 2^w-2^(w-1)-1:-1:2
        push!(openidx, i => 4)
    end
    push!(openidx, 1 => 3)

    for i in 1:2^(w-1)
        pushfirst!(openidx, length(openidx)+1 => 2)
    end
    return cont, openidx
end

function binary_tree(ϕ::AbstractVector)
    N = nqubits(ϕ)

    # TODO: N must be a power of two

    # TODO: support general "qudits"
    d = 2

    ψ = copy(ϕ)
    w = Int(log2(N)+1)
    tensors = []
    ψ = reshape(ψ, (d, d^(N-1)))
    U, S, V = svd(ψ);

    push!(tensors, Array(U'))
    ubond = length(S)
    ψ = diagm(S) * adjoint(V)

    for q in 2:N-1
        ψ = reshape(ψ, (ubond*d, d^(N-q)))
        U, S, V = svd(ψ)
        dbond = length(S)
        ψ = diagm(S) * adjoint(V)
        U = reshape(U, (ubond, d, dbond))
        U = permutedims(U, [1,3,2])
        push!(tensors, U)
        ubond = dbond #down bond is up for the next tensor
    end
    push!(tensors, ψ)

    tensors_def = []
    for i in 1:w-1
        add_layer!(tensors, tensors_def)
        contract_inner_layer!(tensors, tensors_def)
    end

    contractions, openidx = contractions_binary_tree(w)
    return GeneralTensorNetwork(Tensor.(tensors_def), contractions, openidx)
end
