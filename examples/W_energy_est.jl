
using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()
using ASE, JuLIP, Optim, PyPlot
using Optimisers, ReverseDiff
using LineSearches: BackTracking
using LineSearches

using ACEluxpots: Pot_single 

# dataset
function gen_dat()
   eam = JuLIP.Potentials.EAM("./potentials/w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
   set_data!(at, "energy", energy(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:100];

rcut = 5.5 
maxL = 0
totdeg = 5
ord = 2
         
model = construct_models([:W])
ps, st = Lux.setup(rng, model)
calc = Pot_single.LuxCalc(model, ps, st, rcut)
p_vec, _rest = destructure(ps)

# energy loss function 
function E_loss(train, calc, p_vec)
   ps = _rest(p_vec)
   st = calc.st
   Eerr = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      E = Pot_single.lux_energy(at, calc, ps, st)
      Eerr += ( (Eref - E) / Nat)^2
   end
   return Eerr 
end

p0 = zero.(p_vec)
E_loss(train, calc, p0)
ReverseDiff.gradient(p -> E_loss(train, calc, p), p0)
Zygote.gradient(p -> E_loss(train, calc, p), p_vec)[1]

obj_f = x -> E_loss(train, calc, x)
# obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> E_loss(train, calc, p), x))
obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

solver = Optim.BFGS()

res = optimize(obj_f, obj_g!, p0, solver,
              Optim.Options(x_tol = 1e-10, f_tol = 1e-10, g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
pargmin = Optim.minimizer(res)