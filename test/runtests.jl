using ACEluxpots
using Test

@testset "ACEluxpots.jl" begin

    # @testset "Test potential" begin include("test_potential.jl") end
    # @testset "Test calculator" begin include("test_calculator.jl") end
    @testset "Test Profile" begin include("test_profile.jl") end
    # @testset "Test Optimization of W" begin include("test_W_optim.jl") end
    # @testset "Tets Optimization of NiAl" begin include("test_NiAl_optim.jl") end

end
