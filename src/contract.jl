using BenchmarkTools

"""
    contraction_rep(net::TensorNetwork)
Switch tensors to get S representation for contraction optimization
"""
function contract_rep(net::TensorNetwork, optimize::Bool)
    leg_costs = Dict{Int, Int}()
    indexlist = [fill(0, ndims(t)) for t in net.tensors]
    for (i, ts) in enumerate(net.contractions)
        for j in ts.idx
            @assert 1 ≤ j.second ≤ ndims(net.tensors[j.first])
            indexlist[j.first][j.second] = i
            leg_costs[i] = size(net.tensors[j.first])[j.second]
        end
    end
    j = length(net.contractions)
    for (i, oi) in enumerate(net.openidx)
        # tensor leg must not participate in a contraction
        @assert indexlist[oi.first][oi.second] == 0
        # last qubit corresponds to fastest varying index
        indexlist[oi.first][oi.second] =  - i - j
        leg_costs[abs(-i -j)] = size(net.tensors[oi.first])[oi.second]
    end
    # consistency check
    for idx in indexlist, i in idx
        @assert i != 0
    end

    return leg_costs, indexlist
end

"""
    contraction_rep(net::TensorNetwork)
Switch tensors to get S representation for contraction optimization
"""
function contract_rep(net::TensorNetwork)
    indexlist = [fill(0, ndims(t)) for t in net.tensors]
    for (i, ts) in enumerate(net.contractions)
        for j in ts.idx
            @assert 1 ≤ j.second ≤ ndims(net.tensors[j.first])
            indexlist[j.first][j.second] = i
        end
    end
    for (i, oi) in enumerate(net.openidx)
        # tensor leg must not participate in a contraction
        @assert indexlist[oi.first][oi.second] == 0
        # last qubit corresponds to fastest varying index
        indexlist[oi.first][oi.second] = i - length(net.openidx) - 1
    end
    # consistency check
    for idx in indexlist, i in idx
        @assert i != 0
    end

    return indexlist
end

"""
    getBuildCost(freelegs, commonlegs, legCosts, μ_old, μ, newObjectFlags[a][i]||newObjectFlags[b][j], Ta.costToBuild, Tb.costToBuild)
calcualtes cost of building tensor given 2 input tensors
implements algorithm for opt_einsum in arxiv:1304.6112
"""
function getBuildCost(freelegs, commonlegs, legCosts, μ_old, μ, isnew, cost1, cost2)
    alllegs = freelegs .| commonlegs
    newCost = 1
    for (i,leg) in enumerate(alllegs)
        if leg
            newCost *= legCosts[i]
        end
    end
    newCost += cost1 + cost2
    if newCost > μ
        return newCost, false
    end
    if  !isnew
        if newCost <= μ_old
            return Inf, false
        end
    end
    newCost, true
end


"""
    check_contraction!(Sab, Sa, Sb, leg_costs)
checks all combinations of tensors in Sa and Sb. Updates Sab for allowable combinations
implements algorithm for opt_einsum in arxiv:1304.6112
"""

function check_contraction!(S, a, b, legcosts, listindex, newObjectFlags, μs)

    μ_old = μs[1]
    μ_cap = μs[2]
    μ_next = μs[3]
    Sa = S[a]
    Sb = S[b]
    Sab = S[a+b]

    for (i, Ta) in enumerate(Sa)
        for (j, Tb) in enumerate(Sb)
            if any(Ta.tensorFlags .& Tb. tensorFlags)
                continue
            end

            commonlegs = Ta.legFlags .& Tb.legFlags
            freelegs = Ta.legFlags .⊻ Tb.legFlags
            freelegs1 = Ta.legFlags .& freelegs
            freelegs2 = Tb.legFlags .& freelegs
            commonlegsflag = any(commonlegs)

            if !commonlegsflag
                continue
            end

            newCost, isOK = getBuildCost(freelegs, commonlegs, legcosts, μ_old, μ_cap, (newObjectFlags[a][i]||newObjectFlags[b][j]), Ta.costToBuild, Tb.costToBuild)
            if !isOK
                μ_next = min(μ_next, newCost)
                continue
            end

            tensorsInNew = Ta.tensorFlags .| Tb.tensorFlags
            legsInNew = Ta.legFlags .| Tb.legFlags
            isnew = true
            objptr = 0
            for (i, object) in enumerate(Sab)
                isnew = object.tensorFlags != tensorsInNew
                if !isnew
                    objptr = i
                    break
                end
            end

            newsequence = Int[]
            for i = 1:length(commonlegs)
                if commonlegs[i] != 0
                    push!(newsequence, i)
                end
            end
            sequence = vcat(Ta.sequenceToBuild, Tb.sequenceToBuild, newsequence)
            newmaxdim = []
            if isnew
                push!(Sab, ContractionObject(freelegs, tensorsInNew, sequence, newCost, false, 0, 0))
                push!(newObjectFlags[a+b], true)
            else
                if Sab[objptr].costToBuild > newCost
                    Sab[objptr] = ContractionObject(freelegs, tensorsInNew, sequence, newCost, false, 0, 0)
                    newObjectFlags[a+b][objptr] = true
                end
            end
        end
    end

    return μ_old, μ_cap, μ_next
end

"""
    ContractionCombination
Datastructure to represent a possible contraction combination
"""
struct ContractionObject
    legFlags::AbstractVector{Bool}
    tensorFlags::AbstractVector{Bool}
    sequenceToBuild::AbstractVector{Int}
    costToBuild::Int
    isOP::Bool
    OPMaxDim::Int
    allInOP::Int
end

"""
    contract_order(net::TensorNetwork)
simple port of contract_path algorithms from github.com/dgasmith/opt_einsum to julia
implements algorithm for opt_einsum in arxiv:1304.6112
"""

function contract_order(net::TensorNetwork, legcosts::Dict{Int,Int}, indexlist::Array)
    n = length(net.tensors)
    S = AbstractVector{ContractionObject}[]
    newObjectFlags = AbstractVector{Bool}[]
    S1 = ContractionObject[]
    numtensors = length(net.tensors)
    numlegs = length(keys(legcosts))
    for (i,x) in enumerate(indexlist)
        legFlags = fill(false, numlegs)
        legFlags[abs.(x)] .= true
        tensorFlags = fill(false, numtensors)
        tensorFlags[i] = true
        sequenceToBuild = []
        costToBuild = 0
        isOP = false
        OPMaxDim = 0
        allInOP = 0

        push!(S1, ContractionObject(legFlags, tensorFlags, sequenceToBuild, costToBuild, isOP, OPMaxDim, allInOP))
    end

    push!(S, S1)
    push!(newObjectFlags, fill(true, numtensors))

    for i in 2:n
        push!(S, ContractionObject[])
        push!(newObjectFlags, Bool[])
    end

    μ_old = 0
    μ_cap = 1
    μ_next = Inf

    while length(S[end]) == 0
        for c in 2:n
            for d in 1:floor(Int, c//2)
                μ_old, μ_cap, μ_next = check_contraction!(S, d, c-d, legcosts, indexlist, newObjectFlags, (μ_old, μ_cap, μ_next))
            end
        end
        μ_old = μ_cap
        μ_cap = μ_next
        μ_next = Inf

        for i in length(newObjectFlags)
            newObjectFlags[i][:] .= false
        end

    end
    sequence = S[end][1].sequenceToBuild
    cost = S[end][1].costToBuild
    return sequence, cost
end

"""
    contract(net)
Fully contract a given TensorNetwork object.
"""

function contract(net::TensorNetwork; optimize=false)


    # TODO: approximate contraction using SVD splittings
    if optimize
        legcosts, indexlist = contract_rep(net; optimize=optimize)
        sequence, cost = contract_order(net, legcosts, indexlist)
        for i in 1:length(indexlist)
            for j in 1:length(indexlist[i])
                if indexlist[i][j] > 0
                    indexlist[i][j] = findall(x->x==indexlist[i][j], sequence)[1]
                end
            end
        end
        return TensorOperations.ncon([t.data for t in net.tensors], indexlist; order=sequence)
    else
        indexlist = contract_rep(net)
    end

    # for now simply forward the contraction operation to 'ncon'
    return TensorOperations.ncon([t.data for t in net.tensors], indexlist)
end
