using Polynomials4ML: lux, legendre_basis
using EquivariantModels: simple_radial_basis
using Optimisers: destructure
using ACEluxpots: _toState
using ACEbase.Testing: println_slim, print_tf, fdtest

using ACEluxpots, EquivariantModels, Lux, StaticArrays, LinearAlgebra, Zygote, Polynomials4ML

using JuLIP, Combinatorics
using Test, Printf, Random


rng = Random.MersenneTwister()

function grad_test2(f, df, X::AbstractVector; verbose = true)
   F = f(X) 
   ∇F = df(X)
   nX = length(X)
   EE = Matrix(I, (nX, nX))

   verbose && @printf("---------|----------- \n")
   verbose && @printf("    h    | error \n")
   verbose && @printf("---------|----------- \n")
   for h in 0.1.^(0:12)
      gh = [ (f(X + h * EE[:, i]) - F) / h for i = 1:nX ]
      verbose && @printf(" %.1e | %.2e \n", h, norm(gh - ∇F, Inf))
   end
end

## 

# === test configs ===
rcut = 5.5 
species = [:W, :Cu, :Ni, :Fe, :Al]
totdeg = 8
radial = simple_radial_basis(legendre_basis(totdeg))
model = construct_model(species, radial)

ps, st = Lux.setup(rng, model)
p_vec, _rest = destructure(ps)

# set up toy JuLIP.Atoms
at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)
iCu = [5, 12]; iNi = [3, 8]; iAl = [10]; iFe = [6];
cats = AtomicNumber.([:W, :Cu, :Ni, :Fe, :Al])
at.Z[iCu] .= cats[2]; at.Z[iNi] .= cats[3]; at.Z[iAl] .= cats[4]; at.Z[iFe] .= cats[5];
nlist = JuLIP.neighbourlist(at, rcut)
_, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, 1)

# get first atom as center atom
z0  = at.Z[1]
X = _toState(Rs, Zs, z0)
model(X, ps, st)

##

@info("test derivative w.r.t X")
g = Zygote.gradient(X -> model(_toState(X, Zs, z0), ps, st)[1], Rs)[1]

F(Rs) = model(_toState(Rs, Zs, z0), ps, st)[1]
dF(Rs) = Zygote.gradient(X -> model(_toState(X, Zs, z0), ps, st)[1], Rs)[1]

fdtest(F, dF, Rs)

##

@info("test derivative w.r.t parameter")
p = Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
p, = destructure(p)
W0, re = destructure(ps)

Fp = w -> model(X, re(w), st)[1]
dFp = w -> ( gl = Zygote.gradient(p -> model(X, p, st)[1], ps)[1]; destructure(gl)[1])
grad_test2(Fp, dFp, W0)
