using Test
using TestSetExtensions
using Qaintensor
using Qaintessent

@testset ExtendedTestSet "shifts" begin

    step = 5
    p1 = 1 => 1
    p2 = 6 => 1
    p3 = 11 => 1

    @test Qaintensor.shift_pair(p1, step) == p2

    s1 = Summation([p1, p2])
    s2 = Summation([p2, p3])

    @test Qaintensor.shift_summation(s1, step) == s2
end
