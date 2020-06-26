function MCTS.estimate_value(est::BasicPOMCP.SolvedFORollout,
                             bmdp::GenerativeBeliefMDP,
                             belief,
                             d::Int64)
    sim = RolloutSimulator(est.rng, Nullable{Any}(), Nullable{Float64}(), Nullable(d))
    return simulate(sim, bmdp.pomdp, est.policy, rand(est.rng, belief))
end

function MCTS.estimate_value(est::BasicPOMCP.SolvedFOValue{P},
                             bmdp::MeanRewardBeliefMDP,
                             b::AbstractParticleBelief,
                             d::Int64) where P <: ValueIterationPolicy
    r = [estimate_value(est, bmdp.pomdp, s, d) for s in particles(b)]
    mean(r, weights(weights(b)))
end

function MCTS.estimate_value(estimator::BasicPOMCP.SolvedFOValue{P}, pomdp::POMDP, start_state, steps::Int) where P <: ValueIterationPolicy
    POMDPs.value(estimator.policy, start_state)
end

function MCTS.estimate_value(est::BasicPOMCP.SolvedFOValue{P},
                             bmdp::MeanRewardBeliefMDP,
                             b::AbstractParticleBelief,
                             d::Int64) where P <: AlphaVectorPolicy
    estimate_value(est, bmdp.pomdp, b, d)
end

function MCTS.estimate_value(estimator::BasicPOMCP.SolvedFOValue{P}, pomdp::POMDP, b, steps::Int) where P <: AlphaVectorPolicy
    POMDPs.value(estimator.policy, b)
end

Random.seed!(planner::Policy, x) = planner
