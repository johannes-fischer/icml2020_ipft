### Kernel Density Estimation
export
    kde,
    KDEMethod,
    FixedBandwidth,
    SilvermansRule,
    bandwidth
include("kde_methods.jl")

export
    gmm_matrix,
    gmm_matrix!
include("gmm_matrix.jl")


### Information Measure calculation
export
    information,
    InformationMeasure,
    NoInformation
export
    ParticleInformationMeasure,
    DiscreteEntropy,
    discrete_entropy
export
    KDEInformationMeasure,
    KDEEntropy,
    GMMEntropy



abstract type InformationMeasure end
"""
    Measures how much information is contained in the given belief.
    Measuring is done with the given InformationMeasure.
"""
information(m::InformationMeasure, args...; kwargs...)::Float64 = (error("information not implemented for $m"); return 0.0)

struct NoInformation <: InformationMeasure end
information(::NoInformation; kwargs...)::Float64 = 0.0

abstract type ParticleInformationMeasure <: InformationMeasure end

struct DiscreteEntropy <: ParticleInformationMeasure end
information(::DiscreteEntropy; bp::AbstractParticleBelief, kwargs...)::Float64 = -discrete_entropy(bp)
# Works only if enough particles in every discrete state, by summing up the weights
# for continuous distributions where all particles are in different states this yields -log(1/n_particles(b))
discrete_entropy(b::AbstractParticleBelief)::Float64 = -sum(p * log(p) for p in values(ParticleFilters.probdict(b)) if p > 0.0)

abstract type KDEInformationMeasure <: InformationMeasure end

struct GMMEntropy <: KDEInformationMeasure
    kde_method::KDEMethod
end
function information(m::GMMEntropy; bp::AbstractParticleBelief, kwargs...)::Float64
    ws = weight_sum(bp)
    dot(weights(bp)/ws, log.(gmm_matrix(bp, bandwidth(m.kde_method, bp)) * weights(bp)/ws))
end
