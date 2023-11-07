
# can be shorten
using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
rng = Random.MersenneTwister()
using Optimisers, ReverseDiff, Optim
using LineSearches: BackTracking
using LineSearches
using ACEluxpots: construct_model, LuxCalc, lux_efv
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("../../examples/potentials/w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
   set_data!(at, "energy", energy(eam, at))
   set_data!(at, "forces", forces(eam, at))
   set_data!(at, "virial", virial(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:20];

rcut = 5.5 
maxL = 0
totdeg = 3
ord = 2

# === set up model and calculator ===
radial = simple_radial_basis(legendre_basis(totdeg))
model = construct_model([:W], radial)
ps, st = Lux.setup(rng, model)
calc = LuxCalc(model, ps, st, rcut)
p_vec, _rest = destructure(ps)

function loss(train, calc, p_vec)
   p1 = _rest(p_vec)
   st = calc.st
   err = 0
   for at in train
      Nat = length(at)
      Eref = at.data["energy"].data
      Fref = at.data["forces"].data
      Vref = at.data["virial"].data
      E, F, V = lux_efv(at, calc, p1, st)
      err += ( (Eref-E) / Nat)^2 # + 
            # sum( f -> sum(abs2, f), (Fref .- F) ) / Nat #  + 
            # sum(abs2, (Vref.-V) )
   end
   return err
end

# p0 = zero.(p_vec)
# loss(train, calc, p0)
# ReverseDiff.gradient(p -> loss(train, calc, p), p0)

obj_f = x -> loss(train, calc, x)
obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> loss(train, calc, p), x))

solver = Optim.BFGS()

p0 = zero.(p_vec)
res = optimize(obj_f, obj_g!, p0, solver,
               Optim.Options(g_tol = 1e-6, show_trace = true))

# Eerrmin = Optim.minimum(res)
# RMSE = sqrt(Eerrmin / length(train))
# pargmin = Optim.minimizer(res)