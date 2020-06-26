abstract type InformationMCTSPlanner{P} <: AbstractMCTSPlanner{P} end

struct InformationBelief{B <: AbstractParticleBelief}
    b::B
    i::Float64
end

Base.rand(rng::AbstractRNG, ib::InformationBelief) = rand(rng, ib.b)
Statistics.mean(ib::InformationBelief) = mean(ib.b)

ParticleFilters.pdf(ib::InformationBelief, s::S) where {S} = pdf(ib.b,s)

function ParticleFilters.resample(resampler, d::Sampleable, rng::AbstractRNG)
    n = n_init_samples(resampler)
    ParticleCollection([rand(rng, d) for i=1:n])
end
