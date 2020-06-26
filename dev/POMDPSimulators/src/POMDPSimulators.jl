module POMDPSimulators

using POMDPs
using Random
using ProgressMeter
using DataFrames
using POMDPPolicies
using POMDPModelTools
using BeliefUpdaters
using Distributed

using Test

import POMDPs: simulate, discount

export RolloutSimulator
include("rollout.jl")

export
    SimHistory,
    POMDPHistory,
    MDPHistory,
    AbstractPOMDPHistory,
    AbstractMDPHistory,
    HistoryIterator,
    eachstep,
    state_hist,
    action_hist,
    observation_hist,
    belief_hist,
    reward_hist,
    info_hist,
    ainfo_hist,
    uinfo_hist,
    exception,
    backtrace,
    undiscounted_reward,
    discounted_reward,
    n_steps,
    step_tuple
include("history.jl")

export sim
include("sim.jl")

export HistoryRecorder
include("history_recorder.jl")

export
    StepSimulator,
    stepthrough
include("stepthrough.jl")

export
    Sim,
    run,
    run_parallel,
    problem
include("parallel.jl")

end # module
