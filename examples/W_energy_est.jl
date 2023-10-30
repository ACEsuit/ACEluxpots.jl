
using EquivariantModels, Lux, StaticArrays, Random, LinearAlgebra, Zygote 
using Polynomials4ML: LinearLayer, RYlmBasis, lux 
using EquivariantModels: degord2spec, specnlm2spec1p, xx2AA
rng = Random.MersenneTwister()

# dataset
using ASE, JuLIP
function gen_dat()
   eam = JuLIP.Potentials.EAM("w_eam4.fs")
   at = rattle!(bulk(:W, cubic=true) * 2, 0.1)
   set_data!(at, "energy", energy(eam, at))
   return at
end
Random.seed!(0)
train = [gen_dat() for _ = 1:100];

rcut = 5.5 
maxL = 0
Aspec, AAspec = degord2spec(; totaldegree = 6, 
                              order = 3, 
                              Lmax = maxL, )

l_basis, ps_basis, st_basis = equivariant_model(AAspec, maxL)
X = [ @SVector(randn(3)) for i in 1:10 ]
B = l_basis(X, ps_basis, st_basis)[1][1]

# now build another model with a better transform 
L = maximum(b.l for b in Aspec) 
len_BB = length(B) 
get1 = WrappedFunction(t -> t[1])
embed = Parallel(nothing; 
       Rn = Chain(trans = WrappedFunction(xx -> [1/(1+norm(x)) for x in xx]), 
                   poly = l_basis.layers.embed.layers.Rn, ), 
      Ylm = Chain(Ylm = lux(RYlmBasis(L)),  ) )

model = Chain( 
         embed = embed, 
         A = l_basis.layers.A, 
         AA = l_basis.layers.AA, 
         # AA_sort = l_basis.layers.AA_sort, 
         BB = l_basis.layers.BB, 
         get1 = WrappedFunction(t -> t[1]), 
         dot = LinearLayer(len_BB, 1), 
         get2 = WrappedFunction(t -> t[1]), )
ps, st = Lux.setup(rng, model)
out, st = model(X, ps, st)


ps.dot.W[:] .= 0.01 * randn(length(ps.dot.W)) 
calc = Pot.LuxCalc(model, ps, st, rcut)

using Optimisers, ReverseDiff

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

using Optim
obj_f = x -> E_loss(train, calc, x)
# obj_g! = (g, x) -> copyto!(g, ReverseDiff.gradient(p -> E_loss(train, calc, p), x))
obj_g! = (g, x) -> copyto!(g, Zygote.gradient(p -> E_loss(train, calc, p), x)[1])

using LineSearches: BackTracking
using LineSearches
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

using PyPlot
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