using Test
using TestSetExtensions
using Qaintensor
using Qaintessent

@testset ExtendedTestSet "shifts" begin
    step = 5
    p1 = 1 => 1
    p2 = 6 => 1
    p3 = 11 => 1
    @testset "shift_pair" begin
        @test Qaintensor.shift_pair(p1, step) == p2
    end

    @testset "shift_summation" begin
        s1 = Summation([p1, p2])
        s2 = Summation([p2, p3])
        @test Qaintensor.shift_summation(s1, step) == s2
    end
end

@testset "is_power_two" begin
    N = rand(2:10)
    @test Qaintensor.is_power_two(2^N)
    @test !Qaintensor.is_power_two(2^N-1)
end
