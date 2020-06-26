using ParticleFilters
using Parameters
using Distributions
using QMDP
using POMDPPolicies
using StaticArrays
using POMDPModelTools

import POMDPs: max_possible_weight, new_particle, reset_distribution

export
    SimpleLightDark,
    DSimpleLightDark,
    LDHSolver,
    LDSide,
    LDHeuristic,
    LDSidePolicy,

    max_possible_weight,
    new_particle,

    ContinuousLightDark,
    CLDPingPong,
    CLDPingPongPolicy,
    CLDInformationGain,
    CLDGoalPolicy,
    CLDHeuristic

include("simple_ld.jl")
include("continuous_ld.jl")
