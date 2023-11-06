using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote, Polynomials4ML
using Polynomials4ML: LinearLayer, RYlmBasis, lux , legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using DecoratedParticles: State
using ACEluxpots: construct_model, LuxCalc, lux_efv, lux_energy
using ACEbase.Testing: print_tf

using Test
using ACEbase
using Optimisers
using ReverseDiff
using JuLIP


rng = Random.MersenneTwister()

##

# === test configs === 
rcut = 5.5 
maxL = 0
totdeg = 6
ord = 3

##

# === set up model and calculator ===
radial = simple_radial_basis(legendre_basis(totdeg))
model = construct_model([:W], radial)
ps, st = Lux.setup(rng, model)
calc = LuxCalc(model, ps, st, rcut)

p_vec, _rest = destructure(ps)
JuLIP.usethreads!(false)
ps.dot.W[:] .= 1e-2 * randn(length(ps.dot.W)) 

##

@info("Test energy and forces evalution")
for ntest = 1:30
    at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, randn())
    E, F, V = lux_efv(at, calc, ps, st)
    print_tf(@test JuLIP.energy(calc, at) ≈ E && JuLIP.forces(calc, at) ≈ F && JuLIP.virial(calc, at) ≈ V)
end

println()

at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, randn())

@info("Test gradient w.rt. parameter")
lux_energy(at, calc, ps, st)

F1(p) = lux_energy(at, calc, _rest(p), st)
dF1(p) = Zygote.gradient(_p -> lux_energy(at, calc, _rest(_p), st), p)[1]
@assert ACEbase.Testing.fdtest(F1, dF1, p_vec)

@info("Test double pb")
# define dummy loss
loss(calc, p_vec, st) =  norm(x for x in lux_efv(at, calc, _rest(p_vec), st))
ReverseDiff.gradient(p -> loss(calc, p, st), p_vec)[1]

F2(_p) = loss(calc, _p, st)
dF2(_p) = ReverseDiff.gradient(__p -> F2(__p), _p)
@assert ACEbase.Testing.fdtest(F2, dF2, p_vec)