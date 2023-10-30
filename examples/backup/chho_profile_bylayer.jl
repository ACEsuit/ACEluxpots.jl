using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()
using Optimisers, ReverseDiff

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("/zfs/users/jerryho528/jerryho528/julia_ws/EquivariantModels.jl/examples/potential/w_eam4.fs")
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

##

## AA basis
# first filt out those unfeasible spec_nlm
rSH = false
islong = true
L = maxL
d = 3

using EquivariantModels
EQM = EquivariantModels

spec_nlm = AAspec
filter_init = rSH ? RPE_filter_real(L) : (islong ? EQM.RPE_filter_long(L) : EQM.RPE_filter(L))
spec_nlm = spec_nlm[findall(x -> filter_init(x) == 1, spec_nlm)]

# sort!(spec_nlm, by = x -> length(x))
spec_nlm = EQM.closure(spec_nlm,filter_init; categories = [])
luxchain_tmp, ps_tmp, st_tmp = EquivariantModels.xx2AA(spec_nlm, radial; categories = [], d = d, rSH = rSH)

X = [ @SVector(randn(3)) for i in 1:10 ]

# out = luxchain_tmp(X, ps_tmp, st_tmp)[1]
luxchain_tmp.embed

@info("evaluate")
@profview let model = model, X = X, ps = ps, st = st
   for _ = 1:10000
    luxchain_tmp(X, ps_tmp, st_tmp)
    # Zygote.gradient(p -> model(X, p, st)[1], ps)[1]
   end
end

