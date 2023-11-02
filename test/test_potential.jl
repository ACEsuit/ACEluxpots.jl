using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote, Polynomials4ML
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
using JuLIP, Combinatorics, Test
using ACEbase.Testing: println_slim, print_tf, fdtest
using Optimisers: destructure
using Printf

using ACEluxpots

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

rng = Random.MersenneTwister()

rcut = 5.5 
spec = [:W, :Cu, :Ni, :Fe, :Al]

model = construct_models(spec)
ps, st = Lux.setup(rng, model)
p_vec, _rest = destructure(ps)

at = rattle!(bulk(:W, cubic=true, pbc=true) * 2, 0.1)
iCu = [5, 12]; iNi = [3, 8]; iAl = [10]; iFe = [6];
cats = AtomicNumber.([:W, :Cu, :Ni, :Fe, :Al])
at.Z[iCu] .= cats[2]; at.Z[iNi] .= cats[3]; at.Z[iAl] .= cats[4]; at.Z[iFe] .= cats[5];
nlist = JuLIP.neighbourlist(at, rcut)
_, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, 1)
# centere atom
z0  = at.Z[1]
# serialization, I want the input data structure to lux as simple as possible
get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]
Z0S = get_Z0S(z0, Zs)
# input of luxmodel
X = [Rs, Z0S]

model(X, ps, st)

# testing derivative (forces)
g = Zygote.gradient(X -> model(X, ps, st)[1], X)[1]
grad_model(X, ps, st) = Zygote.gradient(_X -> model(_X, ps, st)[1], X)[1]

F(Rs) = model([Rs, Z0S], ps, st)[1]
dF(Rs) = Zygote.gradient(rs -> model([rs, Z0S], ps, st)[1], Rs)[1]

@info("test derivative w.r.t X")
print_tf(@test fdtest(F, dF, Rs; verbose=true))

@info("test derivative w.r.t parameter")
p = Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
p, = destructure(p)

W0, re = destructure(ps)
Fp = w -> model(X, re(w), st)[1]
dFp = w -> ( gl = Zygote.gradient(p -> model(X, p, st)[1], ps)[1]; destructure(gl)[1])
grad_test2(Fp, dFp, W0)
