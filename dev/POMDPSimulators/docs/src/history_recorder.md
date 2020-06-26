# History Recorder

A `HistoryRecorder` runs a simulation and records the trajectory. It returns a [history record](@ref Histories) ([`MDPHistory`](@ref) or [`POMDPHistory`](@ref)).

```julia
hr = HistoryRecorder(max_steps=100)
pomdp = TigerPOMDP()
policy = RandomPolicy(pomdp)

h = simulate(hr, pomdp, policy)
```

More examples can be found in the [POMDPExamples Package](https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/Running-Simulations.ipynb).

```@docs
HistoryRecorder
```
