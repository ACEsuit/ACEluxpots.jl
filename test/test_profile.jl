
using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()
using Optimisers, ReverseDiff
using ASE, JuLIP

using ACEluxpots: Pot_single 
using ACEluxpots
using ProfileView, BenchmarkTools

rcut = 5.5 
maxL = 0
totdeg = 8
ord = 2

model = construct_models([:W])
ps, st = Lux.setup(rng, model)
calc = Pot_single.LuxCalc(model, ps, st, rcut)
p_vec, _rest = destructure(ps)

ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
calc = Pot_single.LuxCalc(model, ps, st, rcut)

X = [ @SVector(randn(3)) for i in 1:10 ]

model(X, ps, st)
Zygote.gradient(X -> model(X, ps, st)[1], X)[1]


Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
ReverseDiff.gradient(p -> loss(train, calc, p), p_vec)

@info("evaluate")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:100
      Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
   end
end

@btime $model($X, $ps, $st)

@info("gradient w.r.t. X")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:100
      Zygote.gradient(X -> model(X, ps, st)[1], X)[1]
   end
end

@info("gradient w.r.t. parameter")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:100
      Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
   end
end

@info("double pb")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:5
      ReverseDiff.gradient(p -> loss(train, calc, p), p_vec)
   end
end