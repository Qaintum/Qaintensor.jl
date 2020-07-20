using Test
using TestSetExtensions
using Qaintensor
using LinearAlgebra

@testset ExtendedTestSet "mpo" begin

    #Test for converting a random matrix into an MPO
    M = 3
    A = rand(ComplexF64, 2^M ,2^M)
    A_MPO = MPO(copy(A))
    @test reshape(contract(A_MPO), (2^M, 2^M)) ≈ A


    function randn_complex(size)
        return (randn(size)
         + im*randn(size)) / sqrt(2)
    end

    Ucnot = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];
    Ucnot13 = reshape(kron(Ucnot, Matrix(I, 2, 2)), (2, 2, 2, 2, 2, 2))
    Ucnot13 = permutedims(Ucnot13, [2,1,3,5,4,6])
    Ucnot13 = reshape(Ucnot13, (8, 8));

    # create random MPS tensors, using bond dimensions (1, 3, 4, 1)
    mps = [randn_complex((2, 3)), randn_complex((3, 2, 4)), randn_complex((4, 2))]
    ψ_mps = ClosedMPS(Tensor.(mps));
    ψ = reshape(contract(ψ_mps), 2^3);

    #Test for CNOT
    ψ_mps_cnot = apply_MPO(ψ_mps, MPO(Ucnot), (1,3));
    @test reshape(contract(ψ_mps_cnot), 8) ≈ Ucnot13*ψ

    #Test for 2-qubit gates acting on non-adjacent qubits
    N, M = 3, 2

    iwire = (1,3)
    wbef = (1:iwire[1]-1...,)
    wmid = setdiff((iwire[1]:iwire[end]...,), iwire )
    waft = (iwire[end]+1:N...,)
    perm = [wbef...; wmid...; iwire...; waft...]
    println(perm)
    U = rand(ComplexF64, 2^M ,2^M)

    ψ_mps_prime = apply_MPO(ψ_mps, MPO(U), (1,3));

    U_MPO = circuit_MPO(MPO(U), iwire)
    U_MPO = reshape(contract(U_MPO), 2^N, 2^N)

    U = reshape(kron(U, Matrix(1I, 2^(N-M), 2^(N-M))), (fill(2, 2N)...,) );
    U = permutedims(U, [perm; perm.+ N])
    U = reshape(U, (2^N, 2^N));

    @test U_MPO ≈ U
    @test reshape(contract(ψ_mps_prime), 8) ≈ U*ψ

end

let N ∈ 𝜡+
twires = 1:N;
M = 3
iwires = (5,2,7);

U = reshape(kron(U, Matrix(1I, 2^(N-M), 2^(N-M))), (fill(2, 2N)...,) );
# BIG complication is that index ordering in Qaintessent runs backwards
# i.e. wire 1 is at position N and wire 2 at N-1 etc.
# This is resolved by mapping wire `i` is at position `j`
# j = (N+1) - i
twires → 1, 2, 3, 4, 5, 6, 7, 8, 9, ... N
iwires → 5, 2, 7, I, I, I, I, I, I (represent I as 0 or -1 in actual implementation)
perm = []
for twire in twires
    i_index = findall(x->x==twire, iwires) # should only give 1 value if iwires has unique positive values and wire `i` is relevant
    if isnothing(i_index)
        # wire not important, should be I
        if iwires[twire] == I
            j_index = (N+1) - twire # no perm, simple case
        else
            i_index = findall(x->x==I, iwires[twire+1:end])[1] # don't permute already correct indices, find next 'free'
            j_index = (N+1) - i_index
            iwires[i_index], iwires[twire] = iwires[twire], iwires[i_index]
        end
    else
        j_index = (N+1) - i_index[1]
        iwires[i_index], iwires[twire] = iwires[twire], iwires[i_index]
    end
    push!(perm, j_index)

end
