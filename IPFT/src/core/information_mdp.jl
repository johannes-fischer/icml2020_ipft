struct InformationRewardBeliefMDP{P<:POMDP, R, S, B, A} <: MDP{InformationBelief{B}, A}
    pomdp::P
    resample::R
    im::InformationMeasure
    discount_information_gain::Bool
    max_frac_replaced::Float64
    rng::AbstractRNG
    _pm::Vector{S}
    _rm::Vector{Float64}
    _wm::Vector{Float64}
end

function InformationRewardBeliefMDP(pomdp::P, resampler, im, discount_information_gain, max_frac_replaced, rng=Random.GLOBAL_RNG) where {P<:POMDP}
    S = statetype(pomdp)
    b0 = resample(resampler, initialstate_distribution(pomdp), rng)
    InformationRewardBeliefMDP{P, typeof(resampler), S, typeof(b0), actiontype(pomdp)}(
        pomdp,
        resampler,
        im,
        discount_information_gain,
        max_frac_replaced,
        rng,
        S[],
        Float64[],
        Float64[]
        )
end

function POMDPs.generate_sr(bmdp::InformationRewardBeliefMDP{P,R,S,B,A}, ib::InformationBelief{B}, a, rng::AbstractRNG) where {P,R,S,B,A}
    b = ib.b

    if n_particles(b) > bmdp.resample.n
        b = resample(bmdp.resample, b, rng)
    end
    s_o = rand(rng, Base.filter(s->!isterminal(bmdp.pomdp, s), particles(b)))
    sp_o, o = generate_so(bmdp.pomdp, s_o, a, rng)
    while true
        sp_o, o = generate_so(bmdp.pomdp, s_o, a, rng)
        obs_weight(bmdp.pomdp, s_o, a, sp_o, o) > eps(1.0/bmdp.resample.n) && break
        # if obs_weight(bmdp.pomdp, s_o, a, sp_o, o) > eps(1.0/bmdp.resample.n)
        #     break
        # end
    end
    obs_w = obs_weight(bmdp.pomdp, s_o, a, sp_o, o)

    ps = particles(b)
    pm = bmdp._pm
    rm = bmdp._rm
    wm = bmdp._wm
    resize!(pm, 0)
    resize!(rm, 0)
    resize!(wm, 0)

    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(bmdp.pomdp, s)
            all_terminal = false
            sp, r = generate_sr(bmdp.pomdp, s, a, rng)
            if sp isa Real && !isfinite(sp)
                @warn "Generated sp is not finite"
                @show sp s a transition(bmdp.pomdp,s,a)
            end
            push!(pm, sp)
            push!(rm, r)
            push!(wm, obs_weight(bmdp.pomdp, s, a, sp, o))
        end
    end
    ws = sum(wm)
    if all_terminal
        @warn("All states in particle collection were terminal.")
        @debug b length(particles(b)) s_o a sp_o o pm wm ws obs_w
        # return (resample(bmdp.resample, reset_distribution(bmdp.pomdp, b, a, o), rng), ib.i), 0.0
        return InformationBelief{B}(resample(bmdp.resample, initialstate_distribution(bmdp.pomdp), rng), ib.i), [0.0, 0.0]
    elseif ws < eps(1.0/length(wm))
        # @warn("Observation weights are low, ws=",ws)
        @debug b length(particles(b)) s_o a sp_o o pm wm ws obs_w
        # return (resample(bmdp.resample, reset_distribution(bmdp.pomdp, b, a, o), rng), ib.i), 0.0
        wm = ones(length(wm))
        ws = sum(wm)
    end

    # Information Measure calculation
    bpred = ParticleCollection(pm)
    wbp = WeightedParticleBelief(pm, wm/ws)
    pc = resample(bmdp.resample, wbp, rng)

    ip = @inferred information(bmdp.im, bp=pc, wbp=wbp, b=b, bpred=bpred, a=a, o=o)
    @assert isfinite(ip)

    ps = particles(pc)
    mpw = max_possible_weight(bmdp.pomdp, a, o)
    frac_replaced = bmdp.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/mpw)
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = new_particle(bmdp.pomdp, b, a, o, rng)
    end

    d = bmdp.discount_information_gain ? discount(bmdp) : 1
    InformationBelief{B}(pc, ip), Array{Float64}([mean(rm,weights(wm)), d * ip - ib.i])
end

# function initialstate(bmdp::InformationRewardBeliefMDP, rng::AbstractRNG)
#     return resample(bmdp.resample, initialstate_distribution(bmdp.pomdp), rng)
# end

POMDPs.actions(bmdp::InformationRewardBeliefMDP{P,R,S,B,A}, ib::InformationBelief{B}) where {P,R,S,B,A} = actions(bmdp.pomdp, ib.b)
POMDPs.actions(bmdp::InformationRewardBeliefMDP) = actions(bmdp.pomdp)

POMDPs.isterminal_act(bmdp::InformationRewardBeliefMDP{P,R,S,B,A}, a::A) where {P,R,S,B,A} = isterminal_act(bmdp.pomdp, a)
POMDPs.isterminal(bmdp::InformationRewardBeliefMDP{P,R,S,B,A}, ib::InformationBelief{B}) where {P,R,S,B,A} = all(isterminal(bmdp.pomdp, s) for s in particles(ib.b))
# POMDPs.isterminal(bmdp::InformationRewardBeliefMDP, ib) = all(isterminal(bmdp.pomdp, s) for s in particles(ib.b))

POMDPs.discount(bmdp::InformationRewardBeliefMDP) = discount(bmdp.pomdp)

function POMDPSimulators.simulate(sim::RolloutSimulator, mdp::InformationRewardBeliefMDP{P,R,S,B,A}, policy::Policy, initialstate::InformationBelief{B}) where {P,R,S,B,A}
    if isterminal(mdp, initialstate)
        return [0.0, 0.0]
    end

    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end

    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initialstate

    disc = 1.0
    r_total = [0.0, 0.0]
    step = 1

    while disc > eps && step <= max_steps
        a = action(policy, s.b)

        sp, r = @inferred generate_sr(mdp, s, a, sim.rng)

        r_total += disc*r

        s = sp

        disc *= discount(mdp)
        step += 1

        if isterminal(mdp, s, a)
            break
        end
    end

    return r_total
end
