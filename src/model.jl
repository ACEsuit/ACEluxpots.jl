
export construct_models

fcut(rcut::Float64, pin::Int=2, pout::Int=2) = r -> (r < rcut ? abs( (r/rcut)^pin - 1)^pout : 0)
ftrans(r0::Float64=2.0, p::Int=2) = r -> ( (1+r0)/(1+r) )^p
construct_radial(totdeg::Int64, rcut::Float64) = simple_radial_basis(legendre_basis(totdeg), fcut(rcut), ftrans())


function construct_models(spec; ord::Int64=2, totdeg::Int64=5, rcut::Float64=5.5, maxL::Int64=0)

    radial = construct_radial(totdeg, rcut)
    _, AAspec = degord2spec(radial; totaldegree = totdeg, order = ord, Lmax = maxL)

    # deal with the categories (will be improved later!)
    if length(spec) != 1
        cats = AtomicNumber.(spec)
        ipairs = collect(Combinatorics.permutations(1:length(cats), 2))
        allcats = collect(SVector{2}.(Combinatorics.permutations(cats, 2)))
        for (i, cat) in enumerate(cats) 
            push!(ipairs, [i, i]) 
            push!(allcats, SVector{2}([cat, cat])) 
        end
        ori_AAspec = deepcopy(AAspec)
        new_AAspec = []
        for bb in ori_AAspec
        newbb = []
        for (t, ip) in zip(bb, ipairs)
            push!(newbb, (t..., s = cats[ip]))
        end
            push!(new_AAspec, newbb)
        end
        luxchain, _, _ = equivariant_model(new_AAspec, radial, maxL; categories=allcats, islong=false)
    else
        luxchain, _, _ = equivariant_model(AAspec, radial, maxL; islong=false)
    end
    
    lB = size(luxchain.layers.BB.op)[1]
    model = append_layers(luxchain, 
                          get1 =  WrappedFunction(t -> real.(t)), 
                          dot = LinearLayer(lB, 1), 
                          get2 = WrappedFunction(t -> t[1]))

    return model
end












# at = train[end]
# nlist = JuLIP.neighbourlist(at, rcut)
# _, Rs, Zs = JuLIP.Potentials.neigsz(nlist, at, 1)
# z0  = at.Z[1]
# get_Z0S(zz0, ZZS) = [SVector{2}(zz0, zzs) for zzs in ZZS]
# Z0S = get_Z0S(z0, Zs)
# X = [Rs, Z0S]
# out, st = luxchain(X, ps, st)

# ps, st = Lux.setup(MersenneTwister(1234), model)

