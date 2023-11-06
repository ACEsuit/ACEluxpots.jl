
using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()
using Optimisers, ReverseDiff
using ASE, JuLIP

using ACEluxpots: Pot_single 
using ACEluxpots
using ProfileView, BenchmarkTools

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("../examples/potentials/w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
   set_data!(at, "energy", energy(eam, at))
   set_data!(at, "forces", forces(eam, at))
   set_data!(at, "virial", virial(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:10];

rcut = 5.5 
maxL = 0
totdeg = 8
ord = 2

model = construct_model([:W])
ps, st = Lux.setup(rng, model)
calc = Pot_single.LuxCalc(model, ps, st, rcut)
p_vec, _rest = destructure(ps)

ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
calc = Pot_single.LuxCalc(model, ps, st, rcut)

X = [ @SVector(randn(3)) for i in 1:10 ]

model(X, ps, st)
Zygote.gradient(X -> model(X, ps, st)[1], X)[1]

function loss(train, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   err = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      Fref = at.data["forces"].data
      Vref = at.data["virial"].data
      E, F, V = Pot_single.lux_efv(at, calc, ps, st)
      err += ( (Eref-E) / Nat)^2 + sum( f -> sum(abs2, f), (Fref .- F) ) / Nat / 100  #  + 
         # sum(abs2, (Vref.-V) )
   end
   return err
end

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