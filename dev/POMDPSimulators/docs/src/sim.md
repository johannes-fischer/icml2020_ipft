# [`sim()`](@id sim-function)

The `sim` function provides a convenient way to interact with a POMDP or MDP environment and return a [history](@ref Histories). The first argument is a function that is called at every time step and takes a state (in the case of an MDP) or an observation (in the case of a POMDP) as the argument and then returns an action. The second argument is a pomdp or mdp. It is intended to be used with Julia's [`do` syntax](https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments-1) as follows:

```julia
pomdp = TigerPOMDP()
history = sim(pomdp, max_steps=10) do obs
    println("Observation was $obs.")
    return TIGER_OPEN_LEFT
end
```
This allows a flexible and general way to interact with a POMDP environment without creating new `Policy` types.

In the POMDP case, an updater can optionally be supplied as an additional positional argument if the policy function works with beliefs rather than directly with observations.

More examples can be found in the [POMDPExamples Package](https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/Running-Simulations.ipynb)

More examples can be found in the [POMDPExamples Package](https://github.com/JuliaPOMDP/POMDPExamples.jl/blob/master/notebooks/Running-Simulations.ipynb)

```@docs
sim
```
