export construct_model

using EquivariantModels: degord2spec, equivariant_model, append_layers
using Polynomials4ML: legendre_basis, LinearLayer
using Lux: WrappedFunction


using Combinatorics

"""
construct_model(species, radial; ord::Int64=2, totdeg::Int64=5, rcut::Float64=5.5, maxL::Int64=0)
Construct a conventational ACE lux potential with categorial (onehot) basis
species: all species of the model
radial: radial basis of the model, can be a lux chain which can be passed to EQM.degord2spec
maxL: maximum level of L included in the model
ord: order of the n-correlation layer, default to 2
totdeg: totaldegree of the model, default to 12
rcut: rcut of the model, default to 5.5
"""
function construct_model(species, radial; ord::Int64=2, totdeg::Int64=12, rcut::Float64=5.5)

    maxL = 0
    _, AAspec = degord2spec(radial; totaldegree = totdeg, order = ord, Lmax = maxL)

    # deal with the categories (will be improved later!)
    if length(species) != 1
        cats = AtomicNumber.(species)
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
