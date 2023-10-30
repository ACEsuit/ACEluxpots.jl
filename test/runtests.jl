using ACEluxpots
using Test

@testset "ACEluxpots.jl" begin

    @testset "Test potential with single-species" begin include("test_potential.jl") end
    @testset "Test potential with multi-species" begin include("test_potential_multi.jl") end
    @testset "Test forces" begin include("test_forces.jl") end

end
