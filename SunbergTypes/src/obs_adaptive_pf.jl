struct ObsAdaptiveParticleFilter{P<:POMDP,S,R,RNG<:AbstractRNG} <: Updater
    pomdp::P
    resample::R
    max_frac_replaced::Float64
    rng::RNG
    _pm::Vector{S}
    _wm::Vector{Float64}
end

function ObsAdaptiveParticleFilter(p::POMDP, resample, max_frac_replaced, rng::AbstractRNG)
    S = statetype(p)
    return ObsAdaptiveParticleFilter(p, resample, max_frac_replaced, rng, S[], Float64[])
end

POMDPs.initialize_belief(up::ObsAdaptiveParticleFilter{S}, d::Any) where S = resample(up.resample, d, up.rng)
POMDPs.update(up::ObsAdaptiveParticleFilter, b, a, o) = update(up, resample(up.resample, b, up.rng), a, o)

function POMDPs.update(up::ObsAdaptiveParticleFilter, b::ParticleFilters.ParticleCollection, a, o)
    if n_particles(b) > 2*up.resample.n
        b = resample(up.resample, b, up.rng)
    end

    ps = particles(b)
    pm = up._pm
    wm = up._wm
    resize!(pm, 0)
    resize!(wm, 0)

    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(up.pomdp, s)
            all_terminal = false
            sp = generate_s(up.pomdp, s, a, up.rng)
            push!(pm, sp)
            od = observation(up.pomdp, s, a, sp)
            push!(wm, pdf(od, o))
        end
    end
    ws = sum(wm)
    if all_terminal || ws < eps(1.0/length(wm))
        # warn("All states in particle collection were terminal.")
        return initialize_belief(up, reset_distribution(up.pomdp, b, a, o))
    end

    pc = resample(up.resample, WeightedParticleBelief{statetype(up.pomdp)}(pm, wm, ws, nothing), up.rng)
    ps = particles(pc)

    mpw = max_possible_weight(up.pomdp, a, o)
    frac_replaced = up.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/mpw)
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(up.rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = new_particle(up.pomdp, b, a, o, up.rng)
    end
    return pc
end

ParticleFilters.weighted_posterior_belief(f::ObsAdaptiveParticleFilter) = WeightedParticleBelief(f._pm, f._wm)
ParticleFilters.predicted_belief(f::ObsAdaptiveParticleFilter) = ParticleCollection(f._pm)
