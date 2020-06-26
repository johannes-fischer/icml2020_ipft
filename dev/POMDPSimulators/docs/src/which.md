# Which Simulator Should I Use?

The simulators in this package provide interaction with simulations of MDP and POMDP environments from a variety of perspectives. Use this page to choose the best simulator to suit your needs.

## I want to run fast rollout simulations and get the discounted reward.

Use the [Rollout Simulator](@ref Rollout).

## I want to evaluate performance with many parallel Monte Carlo simulations.

Use the [Parallel Simulator](@ref Parallel).

## I want to closely examine the histories of states, actions, etc. produced by simulations.

Use the [History Recorder](@ref History-Recorder).

## I want to step through each individual step of a simulation.

Use the [`stepthrough` function](@ref Stepping-through).

## I want to interact with a MDP or POMDP environment from the policy's perspective

Use the [`sim` function](@ref sim-function).

## I want to visualize a simulation.

Visualization is not implemented directly in this package. However, the [Blink POMDP Simulator package](https://github.com/JuliaPOMDP/BlinkPOMDPSimulator.jl) contains a simulator for visualization. Additionally, histories produced by a [`HistoryRecorder`](@ref) or [`sim`](@ref) are can be visualized using the [`render`](https://juliapomdp.github.io/POMDPModelTools.jl/latest/visualization.html#POMDPModelTools.render) function from [POMDPModelTools](https://github.com/JuliaPOMDP/POMDPModelTools.jl).

See the [Visualization Tutorial in POMDPExamples](https://github.com/JuliaPOMDP/POMDPExamples.jl) for more info.
