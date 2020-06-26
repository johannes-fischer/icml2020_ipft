
using DataFrames
using CSV
using Printf
using Random
using Dates

using POMDPs
using ParticleFilters
using MCTS
using POMCPOW
using QMDP
using DiscreteValueIteration
using POMDPSimulators
using POMDPPolicies

using SunbergTypes
using IPFT
using Scenarios

using Plots
pyplot()

seed = Random.GLOBAL_RNG.seed[1]

@show max_time = 1.0
@show max_depth = 30
@show initial_state = -5.0

pomdp = ContinuousLightDark(big_step=3, transition_std=0.1, observation_std_base=0.5)
@show pomdp

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rng = MersenneTwister(seed)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(90.0),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=5.,
                               alpha_observation=1/15.0,
                               estimate_value=RolloutEstimator(RandomSolver(rng)),
                               check_repeat_obs=false,
                               tree_in_info=false,
                               rng=rng
                              )
        solve(solver, pomdp)
    end,

    "ipft" => begin
        rng = MersenneTwister(seed)
        m = 20
        ifm = GMMEntropy(SilvermansRule())
        discount_information_gain = false
        solver = IPFTSolver(n_iterations=typemax(Int),
                        criterion=MultiObjectiveUCB([1.0,60.0],100.0),
                        depth=max_depth,
                        max_time=max_time,
                        check_repeat_action=false,
                        reuse_states=false,
                        k_observation = 5.0,
                        alpha_observation = 1/20,
                        estimate_value=RolloutEstimator(RandomSolver(rng)),
                        enable_action_pw=false,
                        tree_in_info=true,
                        rng=rng
                        )
        belief_mdp = InformationRewardBeliefMDP(pomdp,
                                         LowVarianceResampler(m),
                                         ifm,
                                         discount_information_gain,
                                         0.05
                                        )
        solve(solver, belief_mdp)
    end,
)

histories = []

state_histories=Dict()

for (k, planner) in solvers
    @show k
    state_histories[k] = []

    be_updater = planner isa InformationMCTSPlanner

    Random.seed!(planner, seed+50_000)

    N_p = 10000
    up = ObsAdaptiveParticleFilter(pomdp,
                                       LowVarianceResampler(N_p),
                                       0.05, MersenneTwister(seed+90_000))
    if be_updater
        up = InformationFilter(up, planner.mdp.im)
    end

    # hr = HistoryRecorder(max_steps=100)
    # h = simulate(hr, pomdp, planner, up, initialstate_distribution(pomdp), -5.0)
    # push!(histories, h)
    # display(plot(h.state_hist))
    for (s,a,r,sp,o) in stepthrough(pomdp, planner, up, initialstate_distribution(pomdp), initial_state, "sarspo", max_steps=50)
        @show s, a, r, sp, o
        push!(state_histories[k], s)
    end
    display(plot(state_histories[k]))
end
