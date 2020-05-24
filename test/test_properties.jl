using Test
using TestSetExtensions
using Qaintensor
using LinearAlgebra


@testset ExtendedTestSet "tensor_properties" begin

    n = 4

    A = rand(ComplexF64, n, n)
    U, R = qr(A)
    U = Array(U)
    U_tensor = reshape(U, 2, 2, 2, 2)
    @test isunitary(Tensor(U_tensor), [1,2], [3,4])

    M = rand(ComplexF64, n, n)
    H = Hermitian(M)
    @test ishermitian(Tensor(H))

end
