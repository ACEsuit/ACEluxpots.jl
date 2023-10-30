using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()
using Optimisers, ReverseDiff

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("./potential/w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
   set_data!(at, "energy", energy(eam, at))
   set_data!(at, "forces", forces(eam, at))
   set_data!(at, "virial", virial(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:20];
test = [gen_dat() for _ = 1:20];

## 

# === Model/SitePotential construction ===
rcut = 5.5 
maxL = 0
totdeg = 8
ord = 2

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=2.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = simple_radial_basis(legendre_basis(totdeg),fcut(rcut),ftrans())

Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                              order = ord, 
                              Lmax = maxL, )

l_basis, ps_basis, st_basis = equivariant_model(AAspec, radial, maxL; islong = false)
X = [ @SVector(randn(3)) for i in 1:10 ]
B = l_basis(X, ps_basis, st_basis)[1]

# now extend the above BB basis to a model
len_BB = length(B) 

model = append_layer(l_basis, WrappedFunction(t -> real(t)); l_name=:real)
model = append_layer(model, LinearLayer(len_BB, 1); l_name=:dot)
model = append_layer(model, WrappedFunction(t -> t[1]); l_name=:get1)
         
ps, st = Lux.setup(rng, model)
out, st = model(X, ps, st)

## 

ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
calc = Pot.LuxCalc(model, ps, st, rcut)
at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)

##

## === benchmarks ===
p_vec, _rest = destructure(ps)
model(X, ps, st)
Zygote.gradient(X -> model(X, ps, st)[1], X)[1]


Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
ReverseDiff.gradient(p -> loss(train, calc, p), p_vec)

@info("evaluate")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:100
      # model(X, ps, st)
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




E, F, V = Pot.lux_efv(at, calc, ps, st)
eam = JuLIP.Potentials.EAM("./potential/w_eam4.fs")
Eref = JuLIP.energy(eam, at)
Fref = JuLIP.forces(eam, at)
Vref = JuLIP.virial(eam, at)

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
      err += ( (Eref-E) / Nat)^2 + sum( f -> sum(abs2, f), (Fref .- F) ) / Nat / 30  + 
         sum(abs2, (Vref.-V) ) / 30
   end
   return err
end