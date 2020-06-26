"""
To use RolloutEstimator(policy) where policy is not just a random policy, but a heuristic policy
"""
function MCTS.convert_estimator(ev::RolloutEstimator, solver::IPFTSolver, mdp::InformationRewardBeliefMDP)
    return MCTS.SolvedRolloutEstimator(MCTS.convert_to_policy(ev.solver, mdp.pomdp), solver.rng)
end

"""
Rollout estimator for information reward belief MDP
"""
function MCTS.estimate_value(estimator::BasicPOMCP.SolvedFOValue, bmdp::InformationRewardBeliefMDP, ib::InformationBelief{T}, steps::Int) where {T<:ParticleCollection}
    v = POMDPs.value(estimator.policy, ib.b)
    [v, 0.0]
end
