using POMDPModels
using POMDPs
using POMDPModelTools
using StaticArrays
using Random
using DiscreteValueIteration
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration
using Test

# write your own tests here
include("runtests_versus_discrete_vi.jl")
