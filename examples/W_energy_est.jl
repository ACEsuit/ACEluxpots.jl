
using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux, legendre_basis 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA, simple_radial_basis
rng = Random.MersenneTwister()
using ASE, JuLIP, Optim, PyPlot
using Optimisers, ReverseDiff
using LineSearches: BackTracking
using LineSearches

using ACEluxpots: Pot

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

fcut(rcut::Float64,pin::Int=2,pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=2.0,p::Int=2) = r -> ( (1+r0)/(1+r) )^p
radial = simple_radial_basis(legendre_basis(totdeg),fcut(rcut),ftrans())

Aspec, AAspec = degord2spec(radial; totaldegree = totdeg, 
                              order = ord, 
                              Lmax = maxL, )

l_basis, ps_basis, st_basis = equivariant_model(AAspec, radial, maxL; islong = false)
X = [ @SVector(randn(3)) for i in 1:10 ]
B = l_basis(X, ps_basis, st_basis)[1]

# now build another model with a better transform 
L = maximum(b.l for b in Aspec) 
len_BB = length(B) 
get1 = WrappedFunction(t -> t[1])
embed = Parallel(nothing; 
       Rn = Chain(trans = WrappedFunction(xx -> [1/(1+norm(x)) for x in xx]), 
                   poly = l_basis.layers.embed.layers.Rn, ), 
      Ylm = Chain(Ylm = lux(RYlmBasis(L)),  ) )

len_BB = length(B) 

model = append_layer(l_basis, WrappedFunction(t -> real(t)); l_name=:real)
model = append_layer(model, LinearLayer(len_BB, 1); l_name=:dot)
model = append_layer(model, WrappedFunction(t -> t[1]); l_name=:get1)
         
ps, st = Lux.setup(rng, model)
out, st = model(X, ps, st)


ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
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

p0 = zero.(p_vec)
E_loss(train, calc, p0)
ReverseDiff.gradient(p -> E_loss(train, calc, p), p0)
Zygote.gradient(p -> E_loss(train, calc, p), p_vec)[1]

obj_f = x -> E_loss(train, calc, x)
# obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> E_loss(train, calc, p), x))
obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

# solver = Optim.ConjugateGradient(linesearch = BackTracking(order=2, maxstep=Inf))
# solver = Optim.GradientDescent(linesearch = BackTracking(order=2, maxstep=Inf) )
solver = Optim.BFGS()
# solver = Optim.LBFGS() #alphaguess = LineSearches.InitialHagerZhang(),
               # linesearch = BackTracking(order=2, maxstep=Inf) )

res = optimize(obj_f, obj_g!, p0, solver,
              Optim.Options(x_tol = 1e-10, f_tol = 1e-10, g_tol = 1e-6, show_trace = true))

Eerrmin = Optim.minimum(res)
pargmin = Optim.minimizer(res)

ace = Pot.LuxCalc(model, pargmin, st, rcut)
Eref = []
Eace = []
for tr in train
    exact = tr.data["energy"].data
    estim = Pot.lux_energy(tr, ace, _rest(pargmin), st) 
    push!(Eref, exact)
    push!(Eace, estim)
end

test = [gen_dat() for _ = 1:300];
Eref_te = []
Eace_te = []
for te in test
    exact = te.data["energy"].data
    estim = Pot.lux_energy(te, ace, _rest(pargmin), st) 
    push!(Eref_te, exact)
    push!(Eace_te, estim)
end

figure()
scatter(Eref, Eace, c="red", alpha=0.4)
scatter(Eref_te, Eace_te, c="blue", alpha=0.4)
plot(-142.3:0.01:-141.5, -142.3:0.01:-141.5, lw=2, c="k", ls="--")
PyPlot.legend(["Train", "Test"], fontsize=14, loc=2);
xlabel("Reference energy")
ylabel("ACE energy")
axis("square")
xlim([-142.3, -141.5])
ylim([-142.3, -141.5])
PyPlot.savefig("/figure/W_energy_fitting.png")