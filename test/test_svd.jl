using Test
using TestSetExtensions
using Qaintensor
using LinearAlgebra
using Random

@testset ExtendedTestSet "contract_svd" begin
    t1 = rand(ComplexF64, 4,4)
    t2 = rand(ComplexF64, 4,4)
    t = contract_svd(Tensor(t1),Tensor(t2), (2,1))
    @test t.data ≈ t1*t2

    t1 = Tensor(rand(ComplexF64, 2,3,4,6))
    t2 = Tensor(rand(ComplexF64, 1,5,2,4))

    con1 = Summation([1 => 1, 2 => 3])
    con2 = Summation([1 => 3, 2 => 4])
    open1 = [1 => 2, 1 => 3, 1 => 4, 2 => 1, 2 => 2, 2 => 4]
    open2 = [1 => 1, 1 => 2, 1 => 4, 2 => 1, 2 => 2, 2 => 3]

    tn1 = GeneralTensorNetwork([t1,t2], [con1], open1)
    tn2 = GeneralTensorNetwork([t1,t2], [con2], open2)

    @test contract(tn1) ≈ contract_svd(t1, t2, (1,3)).data
    @test contract(tn2) ≈ contract_svd(t1, t2, (3,4)).data
end

@testset ExtendedTestSet "contract_svd exceptions" begin
    t1 = rand(ComplexF64, 4,4)
    t2 = rand(ComplexF64, 4,4)
    t = contract_svd(Tensor(t1),Tensor(t2), (2,1))
    @test t.data ≈ t1*t2

    t1 = Tensor(rand(ComplexF64, 2,3,4,2))
    t2 = Tensor(rand(ComplexF64, 1,5,2,4))

    con1 = Summation([1 => 1, 2 => 3])
    con2 = Summation([1 => 3, 2 => 4])
    open1 = [1 => 2, 1 => 3, 1 => 4, 2 => 1, 2 => 2, 2 => 4]
    open2 = [1 => 1, 1 => 2, 1 => 4, 2 => 1, 2 => 2, 2 => 3]

    tn1 = GeneralTensorNetwork([t1,t2], [con1], open1)
    tn2 = GeneralTensorNetwork([t1,t2], [con2], open2)

    @test_throws ErrorException("Error must be positive") contract(tn1) ≈ contract_svd(t1, t2, (1,3); er=-0.3).data

    t2_short = Tensor(rand(ComplexF64, 8,8))
    @test_throws ErrorException("Dimensions of contraction legs do not match") contract(tn1) ≈ contract_svd(t1, t2_short, (1,3)).data
    @test contract(tn2) ≈ contract_svd(t1, t2, (3,4)).data
end
