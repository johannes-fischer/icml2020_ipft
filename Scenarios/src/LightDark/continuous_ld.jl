@with_kw struct ContinuousLightDark <: POMDPs.POMDP{Float64,Int,Float64}
    discount_factor::Float64        = 0.95
    correct_r::Float64              = 100.0
    incorrect_r::Float64            = -100.0
    light_loc::Float64              = 10.0
    big_step::Int                   = 3
    movement_cost::Float64          = -1.0
    transition_std::Float64         = 1.0
    observation_std::Float64        = sqrt(2)
    observation_std_base::Float64   = 0.5
    goal_radius::Float64            = 1.0
    initialstate_distribution      = Normal(0.0, 10.0)
end
discount(p::ContinuousLightDark) = p.discount_factor
isterminal_act(::ContinuousLightDark, a::Int) = a == 0

actions(p::ContinuousLightDark) = [-p.big_step, -1, 0, 1, p.big_step]
n_actions(p::ContinuousLightDark) = length(actions(p))
actionindex(p::ContinuousLightDark, a::Int) = findfirst(actions(p), a)

transition(p::ContinuousLightDark, s::Float64, a::Int) = Normal(s+a, p.transition_std)

function observation(p::ContinuousLightDark, sp::Float64)
    Normal(sp, p.observation_std*abs(sp-p.light_loc) + p.observation_std_base)
end

function reward(p::ContinuousLightDark, s::Float64, a::Int, sp::Float64)
    if a == 0
        if abs(s) < p.goal_radius
            return p.correct_r
        elseif abs(s) < 1.5 * p.goal_radius
            return p.correct_r + (p.correct_r-p.incorrect_r)/2 * (p.goal_radius - abs(s))
        elseif abs(s) < 2.0 * p.goal_radius
            return p.incorrect_r + (p.correct_r-p.incorrect_r)*3/2 * (2.0 * p.goal_radius - abs(s))
        else
            return p.incorrect_r
        end
    else
        return p.movement_cost
    end
end

function initialstate_distribution(pomdp::ContinuousLightDark)
    return pomdp.initialstate_distribution
end

# Added to use PFT algorithm

sampletype(d::Normal{T}) where {T} = T

function max_possible_weight(pomdp::ContinuousLightDark, a, o)
    return obs_weight(pomdp, o, o)
end

function new_particle(p::ContinuousLightDark, b, a, o, rng)
    return rand(rng, observation(p, o))
end

max_entropy_distribution(pomdp::ContinuousLightDark) = Normal(0., 100.)


# For LocalApproximationValueIteration
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::Float64, pomdp::ContinuousLightDark)
    v = SVector{1,Float64}(s)
end

function POMDPs.convert_s(::Type{Float64}, v::AbstractVector{Float64}, pomdp::ContinuousLightDark)
    s = v[1]
end

# Walk towards a goal with big and small steps

mutable struct CLDGoalPolicy <: Policy
    pomdp::ContinuousLightDark
    goal::Float64
end

function action(p::CLDGoalPolicy, b)
    m = mean(b)
    small_step = 1
    if abs(m - p.goal) >= (p.pomdp.big_step + 1)/2
        return -p.pomdp.big_step*Int(sign(m-p.goal))
    else
        return -small_step*Int(sign(m-p.goal))
    end
end

# Simple Policy

@with_kw mutable struct CLDPingPong <: Solver
    turnpoint::Float64  = 10.
    direction::Int      = 1
end

struct CLDPingPongPolicy <: Policy
    s::CLDPingPong
end

solve(solver::CLDPingPong, pomdp::ContinuousLightDark) = CLDPingPongPolicy(solver)
Random.seed!(p::CLDPingPongPolicy, s) = p

function action(policy::CLDPingPongPolicy, b)
    if mean(b) > policy.s.turnpoint
        policy.s.direction = -1
    elseif mean(b) < -policy.s.turnpoint
        policy.s.direction = 1
    end
    policy.s.direction
end

# Information greedy Policy

mutable struct CLDInformationGain <: Solver end

solve(solver::CLDInformationGain, pomdp::ContinuousLightDark) = CLDGoalPolicy(pomdp, pomdp.light_loc)
Random.seed!(p::CLDGoalPolicy, s) = p


# Heuristic policy based on standard deviation

mutable struct CLDHeuristic <: Policy
    p::ContinuousLightDark
    policy::CLDGoalPolicy
    std_thresh::Float64
    localized::Bool
end

solve(sol::LDHSolver, pomdp::ContinuousLightDark) = CLDHeuristic(pomdp, CLDGoalPolicy(pomdp, 0.0), sol.std_thresh, false)

function action(p::CLDHeuristic, b::AbstractParticleBelief)
    big_step = p.p.big_step
    small_step = 1
    s = std(b)
    m = mean(b)

    if !p.localized && s <= p.std_thresh
        p.localized = true
    end

    if p.localized
        # If localized set origin as goal
        p.policy.goal = 0.0
        # If close enough tag goal with action 0
        if abs(m - p.policy.goal) < small_step
            return 0
        end
    else
        # Else set light source as goal
        p.policy.goal = p.p.light_loc
    end
    # Move to goal if not close enough
    return action(p.policy, b)
end

action(p::CLDHeuristic, s::Float64) = (p.policy.goal = 0.0; action(p.policy, ParticleCollection([s])))

Base.copy(p::CLDHeuristic) = CLDHeuristic(p.p, p.policy, p.std_thresh, p.localized)
