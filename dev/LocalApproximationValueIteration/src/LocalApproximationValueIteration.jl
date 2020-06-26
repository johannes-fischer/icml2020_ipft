"""
This module implements Value Iteration for large/continuous state spaces by solving for some 
subspace of the state space and interpolating the value function over the rest of the state space
"""
module LocalApproximationValueIteration

using Random
using Printf
using POMDPs
using POMDPModelTools
using LocalFunctionApproximation

import POMDPs: Solver, solve, Policy, action, value


# Exports related to solver
export
    LocalApproximationValueIterationPolicy,
    LocalApproximationValueIterationSolver,
    solve,
    action,
    value,
    action_value

include("local_approximation_vi.jl")
    
end # module