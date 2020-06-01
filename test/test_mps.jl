using Test
using TestSetExtensions
using Qaintensor


@testset ExtendedTestSet "mps" begin

    T = Tensor(rand(2,2,2))
    mps = OpenMPS(T, 3)

    mps_contract=contract(mps)
    mps_svd=contract_svd_mps(mps, 0.0);

    @test mps_contract ≈ mps_svd.data

end
