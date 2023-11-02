

using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote, Polynomials4ML
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()

##
using ACEluxpots: Pot_single

rcut = 5.5 
maxL = 0
totdeg = 6
ord = 3

model = construct_models([:W])
ps, st = Lux.setup(rng, model)
calc = Pot_single.LuxCalc(model, ps, st, rcut)
p_vec, _rest = destructure(ps)

using JuLIP
JuLIP.usethreads!(false) 
ps.dot.W[:] .= 1e-2 * randn(length(ps.dot.W)) 

at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)

@time JuLIP.energy(calc, at)
@time Pot_single.lux_energy(at, calc, ps, st)
@time JuLIP.forces(calc, at)
@time JuLIP.virial(calc, at)

##
using Optimisers, ReverseDiff

p_vec, _rest = destructure(ps)
f(_pvec) = Pot_single.lux_energy(at, calc, _rest(_pvec), st)

f(p_vec)
gz = Zygote.gradient(f, p_vec)[1]

@time f(p_vec)
@time Zygote.gradient(f, p_vec)[1]

# We can use either Zygote or ReverseDiff for gradients. 
gr = ReverseDiff.gradient(f, p_vec)
@show gr ≈ gz

@info("Interestingly ReverseDiff is much faster here, almost optimal")
@time f(p_vec)
@time Zygote.gradient(f, p_vec)[1]
@time ReverseDiff.gradient(f, p_vec)

##
@info("Compute Energies, Forces and Virials at the same time")
E, Force, V = Pot_single.lux_efv(at, calc, ps, st)
@show E ≈ JuLIP.energy(calc, at)
@show Force ≈ JuLIP.forces(calc, at)
@show V ≈ JuLIP.virial(calc, at)
