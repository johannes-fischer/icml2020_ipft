@with_kw struct SimpleLightDark <: POMDPs.POMDP{Int,Int,Float64}
    discount::Float64       = 0.95
    correct_r::Float64      = 100.0
    incorrect_r::Float64    = -100.0
    light_loc::Int          = 10
    radius::Int             = 60
    big_step::Int           = 3
    min_obs_noise::Float64  = 0.5
end
discount(p::SimpleLightDark) = p.discount
isterminal_act(p::SimpleLightDark, a::Int) = a == 0

actions(p::SimpleLightDark) = [-p.big_step, -1, 0, 1, p.big_step]
n_actions(p::SimpleLightDark) = length(actions(p))
actionindex(p::SimpleLightDark, a::Int) = findfirst(actions(p) .== a)

states(p::SimpleLightDark) = -p.radius:p.radius
n_states(p::SimpleLightDark) = length(states(p))
stateindex(p::SimpleLightDark, s::Int) = s+p.radius+1

function transition(p::SimpleLightDark, s::Int, a::Int)
    return SparseCat(SVector(clamp(s+a, -p.radius, p.radius)), SVector(1.0))
end

function observation(p::SimpleLightDark, sp)
    Normal(sp, abs(sp - p.light_loc) + p.min_obs_noise)
end

function reward(p::SimpleLightDark, s, a)
    if a == 0
        return s == 0 ? p.correct_r : p.incorrect_r
    else
        return -1.0
    end
end

function initialstate_distribution(p::SimpleLightDark)
    ps = ones(2*div(p.radius,2)+1)
    ps /= length(ps)
    return SparseCat(div(-p.radius,2):div(p.radius,2), ps)
end

function max_possible_weight(pomdp::SimpleLightDark, a, o)
    return pdf(observation(pomdp, o), o)
end

function new_particle(p::SimpleLightDark, b, a, o, rng)
    return clamp(round(Int, rand(rng, observation(p, o))), -p.radius, p.radius)
end

max_entropy_distribution(pomdp::SimpleLightDark) = DiscreteUniform(-pomdp.radius, pomdp.radius)

@with_kw struct DSimpleLightDark <: POMDPs.POMDP{Int, Int, Int}
    sld::SimpleLightDark = SimpleLightDark()
    binsize::Float64     = 1.0
end

generate_o(p::DSimpleLightDark, sp, rng::AbstractRNG) = floor(Int, rand(rng, observation(p.sld, sp))/p.binsize)
generate_o(p::DSimpleLightDark, a, sp, rng::AbstractRNG) = generate_o(p, sp, rng)
function generate_sor(p::DSimpleLightDark, s, a, rng::AbstractRNG)
    sp = generate_s(p, s, a, rng)
    o = generate_o(p, sp, rng)
    r = reward(p, s, a, sp)
    return sp, o, r
end

function ParticleFilters.obs_weight(p::DSimpleLightDark, sp::Int, o::Int)
    cod = observation(p.sld, sp)
    return cdf(cod, (o+1)*p.binsize) - cdf(cod, o*p.binsize)
end

discount(p::DSimpleLightDark) = discount(p.sld)
isterminal_act(p::DSimpleLightDark, a::Int) = isterminal_act(p.sld, a)
isterminal(p::DSimpleLightDark, s::Number) = isterminal(p.sld, s)
actions(p::DSimpleLightDark) = actions(p.sld)
n_actions(p::DSimpleLightDark) = n_actions(p.sld)
actionindex(p::DSimpleLightDark, a::Int) = actionindex(p.sld, a)
states(p::DSimpleLightDark) = states(p.sld)
n_states(p::DSimpleLightDark) = n_states(p.sld)
stateindex(p::DSimpleLightDark, s::Int) = stateindex(p.sld, s)
transition(p::DSimpleLightDark, s::Int, a::Int) = transition(p.sld, s, a)
reward(p::DSimpleLightDark, s, a) = reward(p.sld, s, a)
initialstate_distribution(p::DSimpleLightDark) = initialstate_distribution(p.sld)

struct LDHeuristic{LD} <: Policy
    p::LD
    q::AlphaVectorPolicy{LD, Int}
    std_thresh::Float64
end

struct LDHSolver <: Solver
    q::QMDPSolver
    std_thresh::Float64
end

LDHSolver(;std_thresh::Float64=0.1, kwargs...) = LDHSolver(QMDPSolver(;kwargs...), std_thresh)

solve(sol::LDHSolver, pomdp::Union{SimpleLightDark}) = LDHeuristic(pomdp, solve(sol.q, pomdp), sol.std_thresh)
solve(sol::LDHSolver, pomdp::DSimpleLightDark) = solve(sol, pomdp.sld)

action(p::LDHeuristic, s::Int) = action(p.q, ParticleCollection([s]))
Random.seed!(p::LDHeuristic, s) = p

function action(p::LDHeuristic, b::AbstractParticleBelief)
    big_step = p.p.big_step
    s = std(b)
    if s <= p.std_thresh
        return action(p.q, b)
    else
        m = mean(particles(b))
        ll = p.p.light_loc
        if m == ll
            return -1*Int(sign(ll))
        elseif abs(m-ll) >= big_step
            return -Int(big_step*sign(m-ll))
        else
            return -Int(sign(m-ll))
        end
    end
end

struct LDSide <: Solver end

mutable struct LDSidePolicy{LD} <: Policy
    q::AlphaVectorPolicy{LD, Int}
end

solve(solver::LDSide, pomdp::Union{SimpleLightDark,DSimpleLightDark}) = LDSidePolicy(solve(QMDPSolver(), pomdp))
Random.seed!(p::LDSidePolicy, s) = p

function action(p::LDSidePolicy, b)
    big_step = p.q.pomdp.big_step
    if pdf(b, mode(b)) > 0.9
        return action(p.q, b)
    else
        return big_step
    end
end
