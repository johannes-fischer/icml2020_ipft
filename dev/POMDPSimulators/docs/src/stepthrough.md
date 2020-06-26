# Stepping through

The [`stepthrough`](@ref) function exposes a simulation as an iterator so that the steps can be iterated through with a for loop syntax as follows:

```julia
pomdp = BabyPOMDP()
policy = RandomPolicy(pomdp)

for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps=10)
    println("in state $s")
    println("took action $o")
    println("received observation $o and reward $r")
end
```

More examples can be found in the [POMDPExamples Package](https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/Running-Simulations.ipynb).

```@docs
stepthrough
```

The `StepSimulator` contained in this file can provide the same functionality with the following syntax:
```julia
sim = StepSimulator("s,a,r,sp")
for (s,a,r,sp) in simulate(sim, problem, policy)
    # do something
end
```
