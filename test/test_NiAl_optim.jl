using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, ACEbase
using Optimisers, ReverseDiff, Optim

using ACEluxpots: Pot

train = read_extxyz("../examples/data/NiAl_data.xyz")
spec = [:Ni, :Al]
rng = Random.MersenneTwister()

rcut = 5.5 
maxL = 0
totdeg = 5
ord = 2

model = construct_models(spec)

ps, st = Lux.setup(MersenneTwister(1234), model)
p_vec, _rest = destructure(ps)

calc = Pot.LuxCalc(model, ps, st, rcut)

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

obj_f = x -> loss(train, calc, x)
obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> loss(train, calc, p), x))
# obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> loss(train, calc, p), x)[1])

p0 = zero.(p_vec)
res = optimize(obj_f, obj_g!, p0,
              Optim.BFGS(),
              Optim.Options(g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
RMSE = sqrt(Eerrmin / length(train))
pargmin = Optim.minimizer(res)

