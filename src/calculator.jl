using JuLIP: Atoms
using StaticArrays: SVector, SMatrix
using Optimisers: destructure
using ChainRulesCore: ignore_derivatives, @ignore_derivatives, NoTangent
using DecoratedParticles: State
using JuLIP, Zygote, StaticArrays

import JuLIP: cutoff
import ACEbase: evaluate!, evaluate_d!

"""
LuxCalc <: JuLIP.SitePotential
Calculator for Lux Potentials
luxmode: Lux model, usually a Lux.Chain
ps: parameter of luxmodel
st: st of luxmodel
rcut: cutoff radius for JuLIP neighbourlist
restructure: a function to reconstruct ps, from Optimisers
"""
struct LuxCalc <: JuLIP.SitePotential 
   luxmodel
   ps::NamedTuple
   st::NamedTuple
   rcut::Float64
   restructure
end

# === constructor ===
function LuxCalc(luxmodel, ps, st, rcut) 
   pvec, rest = destructure(ps)
   return LuxCalc(luxmodel, ps, st, rcut, rest)
end

##

# === utils ===
cutoff(calc::LuxCalc) = calc.rcut

"""
_toState(Rs, Zs, z0)
Convert Rs and Zs from JuLIP neighbourlist to `State`, z0 is the center atom
"""
_toState(Rs, Zs, z0) = [State(rr = ri, Zi = z0, Zj = zj) for (ri, zj) in zip(Rs, Zs)]

import ChainRulesCore: rrule
function rrule(::typeof(_toState), Rs, Zs, z0)
   return [State(rr = ri, Zi = z0, Zj = zj) for (ri, zj) in zip(Rs, Zs)], ∂ -> (NoTangent(), [SVector(∂i.x.rr...) for ∂i in ∂], NoTangent(), NoTangent())
end

##

# === evaluation interface with JuLIP ===
function evaluate!(tmp, calc::LuxCalc, Rs, Zs, z0)
   E, st = calc.luxmodel(_toState(Rs, Zs, z0), calc.ps, calc.st)
   return E[1]
end

function evaluate_d!(dEs, tmpd, calc::LuxCalc, Rs, Zs, z0)
   g = Zygote.gradient(X -> calc.luxmodel(_toState(X, Zs, z0), calc.ps, calc.st)[1], Rs)[1]
   @assert length(g) == length(Rs) <= length(dEs)
   dEs[1:length(g)] .= g 
   return dEs 
end

##

# === parameter estimation stuff  ===
function lux_energy(at::Atoms, calc::LuxCalc, ps::NamedTuple, st::NamedTuple)
   nlist = ignore_derivatives() do 
      JuLIP.neighbourlist(at, calc.rcut)
   end
   return sum( i -> begin
         Js, Rs, Zs = ignore_derivatives() do 
            JuLIP.Potentials.neigsz(nlist, at, i)
         end
         Ei, st = calc.luxmodel(_toState(Rs, Zs, at.Z[i]), ps, st)
         Ei[1] 
      end, 
      1:length(at)
      )
end

function lux_efv(at::Atoms, calc::LuxCalc, ps::NamedTuple, st::NamedTuple)
   nlist = ignore_derivatives() do 
      JuLIP.neighbourlist(at, calc.rcut)
   end
   T = promote_type(eltype(at.X[1]), eltype(ps.dot.W))
   E = 0.0 
   F = zeros(SVector{3, T}, length(at))
   V = zero(SMatrix{3, 3, T}) 
   for i = 1:length(at) 
      Js, Rs, Zs = ignore_derivatives() do 
         JuLIP.Potentials.neigsz(nlist, at, i)
      end

      comp = Zygote.withgradient(X -> calc.luxmodel(_toState(X, Zs, at.Z[i]), ps, st)[1], Rs)
      Ei = comp.val 
      _∇Ei = comp.grad[1]
      ∇Ei = _∇Ei
      # energy 
      E += Ei 

      # Forces 
      for j = 1:length(Rs) 
         F[Js[j]] -= ∇Ei[j] 
         F[i] += ∇Ei[j] 
      end

      # Virial 
      if length(Rs) > 0 
         V -= sum(∇Eij * Rij' for (∇Eij, Rij) in zip(∇Ei, Rs))
      end
   end
   
   return E, F, V 
end

##