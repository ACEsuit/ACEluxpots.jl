using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, ACEbase
using Optimisers, ReverseDiff

using ACEluxpots: Pot

train = read_extxyz("./data/NiAl_data.xyz")

spec = [:Ni, :Al]

rng = Random.MersenneTwister()

rcut = 5.5 
maxL = 0
totdeg = 5
ord = 2

model = construct_models([:Ni, :Al])
ps, st = Lux.setup(MersenneTwister(1234), model)

calc = Pot.LuxCalc(model, ps, st, rcut)

p_vec, _rest = destructure(ps)

# energy loss function 
function E_loss(train, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   Eerr = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      E = Pot.lux_energy(at, calc, ps, st)
      Eerr += ( (Eref - E) / Nat)^2
   end
   return Eerr 
end

function loss(train, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   err = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      Fref = at.data["forces"].data
      Vref = at.data["virial"].data
      E, F, V = Pot.lux_efv(at, calc, ps, st)
      err += ( (Eref-E) / Nat)^2 + sum( f -> sum(abs2, f), (Fref .- F) ) / Nat / 100  #  + 
         # sum(abs2, (Vref.-V) )
   end
   return err
end

p0 = zero.(p_vec)
E_loss(train, calc, p0)
ReverseDiff.gradient(p -> loss(train, calc, p), p0)

using Optim
obj_f = x -> loss(train, calc, x)
obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> loss(train, calc, p), x))
# obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

res = optimize(obj_f, obj_g!, p0,
              Optim.BFGS(),
              Optim.Options(g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
RMSE = sqrt(Eerrmin / length(train))
pargmin = Optim.minimizer(res)

