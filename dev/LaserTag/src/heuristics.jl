# for fully-observable case
struct MoveTowards <: Policy end

function POMDPs.action(p::MoveTowards, s::LTState)
    # try to sneak up diagonally
    diff = s.opponent-s.robot
    dx = sign(diff[1])
    dy = sign(diff[2])
    if abs(dx) != 1 && abs(dy) == 1
        return DIR_TO_ACTION[SVector(dx, dy)]
    else
        if abs(dx) == 1
            dx = 0
        end
        if abs(dy) == 1
            dy = 0
        end
        return DIR_TO_ACTION[SVector(dx, dy)]
    end
end

struct MoveTowardsSampled{RNG} <: Policy
    rng::RNG
end
MoveTowardsSampled() = MoveTowardsSampled(Random.GLOBAL_RNG)

function POMDPs.action(p::MoveTowardsSampled, b)
    s = rand(p.rng, b)
    return action(MoveTowards(), s)
end



struct OptimalMLSolver <: Solver
    solver::Solver
end

struct OptimalML{P<:Policy} <: Policy
    fo_policy::P
end

solve(sol::OptimalMLSolver, p::Union{MDP,POMDP}) = OptimalML(solve(sol.solver, p))
POMDPs.action(pol::OptimalML, b) = action(pol.fo_policy, mode(b))

struct BestExpectedSolver <: Solver
    solver::Solver
end

struct BestExpected{P<:Policy} <: Policy
    fo_policy::P
end

solve(sol::BestExpectedSolver, p::Union{MDP,POMDP}) = BestExpected(solve(sol.solver, p))
function POMDPs.action(pol::BestExpected, b)
    best_eu = -Inf
    best_s = first(iterator(b))
    for s in iterator(b)
        eu = pdf(b, s)*value(pol.fo_policy, s)
        if eu > best_eu
            best_eu = eu
            best_s = s
        end
    end
    return action(pol.fo_policy, best_s)
end
