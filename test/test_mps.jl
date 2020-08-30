using Test
using TestSetExtensions
using Qaintensor
using LinearAlgebra
using Random

@testset ExtendedTestSet "contract_svd_mps" begin
    T = Tensor(rand(2,2,2))
    mps = OpenMPS(T, 3)
    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract = contract(gmps)
    mps_svd = contract_svd_mps(mps; er=0.0)

    @test mps_contract ≈ mps_svd
    @test_throws ErrorException("Function doesn't support periodic boundary conditions for now") contract_svd_mps(PeriodicMPS(T, 3); er=0.0)
    @test_throws ErrorException("Error must be positive") contract_svd_mps(mps; er=-0.5)
end

@testset ExtendedTestSet "mps exceptions" begin
    tensors = [Tensor(rand(2,2,2)), Tensor(rand(2,2,2))]
    openidx = [1=>2, 2=>2]

    contractions = [Summation([1=>1, 2=>3])]
    @test_throws ErrorException("Tensor objects first leg must contract with last leg of previous Tensor object") MPS(tensors, contractions, openidx)

    contractions = [Summation([1=>3, 2=>3])]
    @test_throws ErrorException("Tensor objects last leg must contract with first leg of next Tensor object") MPS(tensors, contractions, openidx )

    tensors = [Tensor(rand(2,2,2,2)), Tensor(rand(2,2,2))]
    contractions = [Summation([1=>4, 2=>1])]
    openidx = [1=>2, 2=>2]
    @test_throws ErrorException("Each Tensor object in MPS form can only have 2 or 3 legs") MPS(tensors, contractions, openidx)
end

@testset ExtendedTestSet "check_mps exceptions" begin
    T = Tensor(rand(2,2,2))
    mps = OpenMPS(T, 3)
    mps.contractions[1] = Summation([1=>2, 2=>1])
    @test_throws ErrorException("Tensor objects first leg must contract with last leg of previous Tensor object") check_mps(mps)

    mps.contractions[1] = Summation([1=>3, 2=>2])
    @test_throws ErrorException("Tensor objects last leg must contract with first leg of next Tensor object") check_mps(mps)

    mps.tensors[2] = Tensor(rand(2,2,2,2))
    @test_throws ErrorException("Each Tensor object in MPS form can only have 2 or 3 legs") check_mps(mps)
end

@testset ExtendedTestSet "conversion from state-vector to mps" begin
    N = 5
    b1 = randn(ComplexF64, (2))
    b2 = randn(ComplexF64, (2))
    b3 = randn(ComplexF64, (2))
    b4 = randn(ComplexF64, (2))
    b5 = randn(ComplexF64, (2))

    ψ_ref = kron(b1,b2,b3,b4,b5)

    mps = MPS(ψ_ref)

    ψ = contract(mps)

    @test ψ_ref ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "conversion from state-vector to mps exceptions" begin
    ψ = randn(ComplexF64, 6)

    @test_throws ErrorException("Input state must have length 2^N") MPS(ψ)
end

@testset ExtendedTestSet "mps constructor" begin
    N = 3

    tensors = Tensor[]
    push!(tensors, Tensor(randn(ComplexF64, (2, 6))))
    push!(tensors, Tensor(randn(ComplexF64, (2, 6, 7))))
    push!(tensors, Tensor(randn(ComplexF64, (2, 7))))

    # contract virtual legs
    contractions = Summation[]
    push!(contractions, Summation([1=>2, 2=>2]))
    push!(contractions, Summation([2=>3, 3=>2]))

    # open (physical) legs
    openidx = Pair[]
    push!(openidx, 1=>1)
    push!(openidx, 2=>1)
    push!(openidx, 3=>1)

    # initial MPS wavefunction
    ψ = GeneralTensorNetwork(tensors, contractions, openidx)

    cgc = qft_circuit(N)

    # conventional statevector representation, as reference
    ψref = apply(cgc, contract(ψ)[:])

    # using tensor network representation
    tensor_circuit!(ψ, cgc)

    @test ψref ≈ contract(ψ)[:]
end

@testset ExtendedTestSet "open_mps" begin

    T = Tensor(rand(2,2,2))
    mps = OpenMPS(T, 3)

    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract = contract(gmps)
    mps_svd = contract_svd_mps(mps; er=0.0)

    @test mps_contract ≈ mps_svd
end

@testset ExtendedTestSet "open_mps exceptions" begin
    tensors = [Tensor(rand(2,2,2,2)), Tensor(rand(2,2,2,2))]
    @test_throws ErrorException("Tensors must have 3 legs") OpenMPS(tensors)
    @test_throws ErrorException("Tensors must have 3 legs") OpenMPS(Tensor(rand(2,2,2,2)), 3)
end

@testset ExtendedTestSet "closed_mps" begin
    T1 = Tensor(rand(2,5))
    T2 = Tensor(rand(5,2,3))
    T3 = Tensor(rand(3,2))
    mps = ClosedMPS([T1, T2, T3])

    gmps = GeneralTensorNetwork(mps.tensors, mps.contractions, mps.openidx)
    mps_contract = contract(gmps)
    mps_svd = contract_svd_mps(mps; er=0.0)

    @test mps_contract ≈ mps_svd
end

@testset ExtendedTestSet "closed_mps exceptions" begin
    T1 = Tensor(rand(2,2,2))
    T2 = Tensor(rand(2,2,2,2))
    T3 = Tensor(rand(2,2))

    @test_throws ErrorException("First tensor must have 2 legs") ClosedMPS([T1, T1, T3])
    @test_throws ErrorException("Tensors must have 3 legs, except the first and last one") ClosedMPS([T3, T2, T3])
    @test_throws ErrorException("Last tensor must have 2 legs") ClosedMPS([T3, T1, T2])
end

@testset ExtendedTestSet "periodic_mps exceptions" begin
    T = Tensor(rand(2,2,2))
    mps = PeriodicMPS(T, 3)
    @test_throws ErrorException contract_svd_mps(mps, er=0.0)
end

@testset ExtendedTestSet "test permute MPS" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end

    order = randperm(N)

    ψ_ref = kron(b...)
    ψ_perm = kron(b[order]...)

    mps = MPS(ψ_ref)
    permute!(mps, order)

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test permute MPS exceptions" begin

    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end

    mps = MPS(kron(b...))

    short_order = [1, 2, 3]
    @test_throws ErrorException("Given permutation must be same length as number of Tensors in MPS") permute!(mps, short_order)

    long_order = [1, 2, 3, 4, 5, 6, 7]
    @test_throws ErrorException("Given permutation must be same length as number of Tensors in MPS") permute!(mps, short_order)

    repeat_order = [1, 1, 1, 1, 1, 1]
    @test_throws ErrorException("Permutation order cannot contain repeat values") permute!(mps, repeat_order)

    negative_order = [1,-2, 3, 4, -5, 6]
    @test_throws ErrorException("Permutation order can only contain positive values") permute!(mps, negative_order)

    negative_order = [1, 2, 8, 4, 5, 6]
    @test_throws ErrorException("Wire numbers in permutation order cannot exceed number of wires in MPS") permute!(mps, negative_order)
end


@testset ExtendedTestSet "test switch MPS qubits" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end
    order = Int[1, 3, 2, 5, 6, 4]
    ψ_ref = kron(b...)
    ψ_perm = kron(b[order]...)

    mps = MPS(ψ_ref)
    switch!(mps, [(2,3), (4,6), (4,5)])

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))

    mps = MPS(ψ_ref)
    switch!(mps, 2, 3)
    switch!(mps, 4, 6)
    switch!(mps, 4, 5)

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test switch MPS exceptions" begin

    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end

    mps = MPS(kron(b...))

    @test_throws ErrorException("Wire indices `i` and `j` must be positive") switch!(deepcopy(mps), 1, -2)
    @test_throws ErrorException("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS") switch!(deepcopy(mps), 7, 4)

    @test_throws ErrorException("Wire indices `i` and `j` must be positive") switch!(deepcopy(mps), [(3, 4), (1, -2)])
    @test_throws ErrorException("Indices to swap `i` and `j` must be less than or equal to the number of open wires in MPS") switch!(deepcopy(mps), [(9, 10), (1, -2)])
end

@testset ExtendedTestSet "test switch neighboring MPS qubits" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end
    order = Int[1, 3, 4, 5, 6, 2]
    ψ_ref = kron(b...)
    ψ_perm = kron(b[order]...)

    mps = MPS(ψ_ref)
    switch!(mps, 4)
    switch!(mps, 3)
    switch!(mps, 2)
    switch!(mps, 1)

    ψ = contract(mps)
    @test ψ_perm ≈ reshape(ψ, (2^N,))
end

@testset ExtendedTestSet "test switch neighboring MPS qubits exceptions" begin
    N = 6
    b = AbstractArray{ComplexF64}[]
    for i in 1:N
        push!(b, randn(ComplexF64, (2)))
    end
    ψ_ref = kron(b...)

    mps = MPS(ψ_ref)

    @test_throws BoundsError switch!(mps, 7)
    @test_throws BoundsError switch!(mps, -2)
end

@testset ExtendedTestSet "copy" begin
    T1 = Tensor(rand(2,5))
    T2 = Tensor(rand(5,2,3))
    T3 = Tensor(rand(3,2))
    mps = ClosedMPS([T1, T2, T3])

    copy_mps = copy(mps)

    @test (copy_mps.tensors == mps.tensors) & (copy_mps.contractions == mps.contractions) & (copy_mps.openidx == mps.openidx)
    
    mps.tensors[1] = Tensor(rand(2,5))
    @test mps.tensors != copy_mps.tensors
end
