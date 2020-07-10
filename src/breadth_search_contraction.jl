using BenchmarkTools

"""
    ContractionObject
Datastructure to represent a possible contraction combination
"""
struct ContractionObject
    legFlags::BitArray{1}
    tensorFlags::BitArray{1}
    sequenceToBuild::Vector{Int}
    costToBuild::Int
    isOP::Bool
    OPMaxDim::Int
    allInOP::Int
end

"""
    leg_cost(all_legs)

calculates cost of a single contraction step
implements algorithm for opt_einsum in arxiv:1304.6112
"""
function leg_cost(all_legs::BitArray{1}, legcosts::Dict{Int, Int})
    newcost = 1
    for (i,leg) in enumerate(all_legs)
        if leg
            newcost *= legcosts[i]
        end
    end
    2*newcost
end

"""
    get_build_cost(freelegs, commonlegs, legCosts, μ_old, μ, newObjectFlags[a][i]||newObjectFlags[b][j], Ta.costToBuild, Tb.costToBuild)

calcualtes cost of building tensor given 2 input tensors
implements algorithm for opt_einsum in arxiv:1304.6112
"""
@inline function get_build_cost(freelegs::BitArray{1}, commonlegs::BitArray{1}, legcosts::Dict{Int,Int}, μ_old::Int, μ::Int, isnew::Bool, cost1::Int, cost2::Int)
    all_legs = freelegs .| commonlegs
    newCost = leg_cost(all_legs, legcosts)
    newCost += cost1 + cost2
    if newCost > μ
        return newCost, false
    end
    if !isnew
        if newCost <= μ_old
            return typemax(Int64), false
        end
    end
    newCost, true
end


"""
    check_contraction!(a::Int, b::Int, Sa::Vector{ContractionObject}, Sb::Vector{ContractionObject}, Sab::Vector{ContractionObject}, legcosts::Dict{Int,Int}, listindex::Vector{Vector{Int}}, newObjectFlags::Vector{Vector{Bool}}, μ_old::Int, μ_cap::Int, μ_next::Int)
checks all combinations of tensors in Sa and Sb. Updates Sab for allowable combinations
implements algorithm for opt_einsum in arxiv:1304.6112
"""

function check_contraction!(a::Int, b::Int, Sa::Vector{ContractionObject}, Sb::Vector{ContractionObject}, Sab::Vector{ContractionObject}, legcosts::Dict{Int,Int}, listindex::Vector{Vector{Int}}, newObjectFlags::Vector{Vector{Bool}}, μ_old::Int, μ_cap::Int, μ_next::Int)
    # Sa = S[a]
    # Sb = S[b]
    # Sab = S[a+b]

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

            newCost, isOK = get_build_cost(freelegs, commonlegs, legcosts, μ_old, μ_cap, (newObjectFlags[a][i]||newObjectFlags[b][j]), Ta.costToBuild, Tb.costToBuild)
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
    contract_order(net::TensorNetwork)
simple port of contract_path algorithms from github.com/dgasmith/opt_einsum to julia
implements algorithm for opt_einsum in arxiv:1304.6112
"""

function optimal_contraction_order(net::TensorNetwork, legcosts::Dict{Int,Int}, indexlist::Vector{Vector{Int}})
    n = length(net.tensors)
    S = Vector{ContractionObject}[]
    newObjectFlags = Vector{Bool}[]
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
    μ_next = typemax(Int64)

    while length(S[end]) == 0
        for c in 2:n
            for d in 1:floor(Int, c//2)
                μ_old, μ_cap, μ_next = check_contraction!(d, c-d, S[d], S[c-d], S[c], legcosts, indexlist, newObjectFlags, μ_old, μ_cap, μ_next)
            end
        end
        μ_old = μ_cap
        μ_cap = μ_next
        μ_next = typemax(Int64)

        for i in length(newObjectFlags)
            newObjectFlags[i][:] .= false
        end

    end
    sequence = S[end][1].sequenceToBuild
    cost = S[end][1].costToBuild
    return sequence, cost
end
