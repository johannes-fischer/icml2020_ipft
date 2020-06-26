abstract type KDEMethod end
kde(m::KDEMethod, args...; kwargs...) = error("kde not implemented for $m, args=$args, kwargs=$kwargs")

struct FixedBandwidth <: KDEMethod
    GMMType
    h::Float64
end
FixedBandwidth(h::Float64) = FixedBandwidth(UnivariateGMM, h)
bandwidth(m::FixedBandwidth, b::AbstractParticleBelief) = m.h

struct SilvermansRule <: KDEMethod
    GMMType
end
SilvermansRule() = SilvermansRule(UnivariateGMM)
bandwidth(m::SilvermansRule, b::AbstractParticleBelief) = max((4/(3*n_particles(b)))^(0.2) * std(b), sqrt(eps()))

kde(m::Union{FixedBandwidth, SilvermansRule}, b::AbstractParticleBelief) = GMMType(b, bandwidth(m, b))
