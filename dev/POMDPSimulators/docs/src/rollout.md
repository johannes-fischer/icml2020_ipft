# Rollout

## RolloutSimulator

`RolloutSimulator` is the simplest MDP or POMDP simulator. When `simulate` is called, it simply simulates a single trajectory of the process and returns the discounted reward.

```julia
rs = RolloutSimulator()
mdp = GridWorld()
policy = RandomPolicy(mdp)

r = simulate(rs, mdp, policy)
```

More examples can be found in the [POMDPExamples Package](https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/Running-Simulations.ipynb)

```@docs
RolloutSimulator
```

