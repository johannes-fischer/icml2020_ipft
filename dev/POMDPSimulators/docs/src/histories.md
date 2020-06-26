# Histories

The results produced by [`HistoryRecorder`](@ref)s and the [`sim`](@ref) function are contained in `SimHistory` objects. A `SimHistory` can be thought of as a colletion of [`NamedTuple`](https://docs.julialang.org/en/v1/manual/types/index.html#Named-Tuple-Types-1)s that each represent a step of the simulation. These named tuples should be accessed using the [`eachstep`](@ref) function.

```@docs
eachstep
```

## Examples:
```julia
collect(eachstep(h, "a,o"))
```
will produce a vector of action-observation named tuples.

```julia
collect(norm(sp-s) for (s,sp) in eachstep(h, "s,sp"))
```
will produce a vector of the distances traveled on each step (assuming the state is a Euclidean vector).

Notes:
- The iteration specification can be specified as a tuple of symbols (e.g. `(:s, :a)`) instead of a string.
- For type stability in performance-critical code, one should construct an iterator directly using `HistoryIterator{typeof(h), (:a,:r)}(h)` rather than `eachstep(h, "ar")`.

`state_hist(h)`, `action_hist(h)`, `observation_hist(h)` `belief_hist(h)`, and `reward_hist(h)` will return vectors of the states, actions, and rewards, and `undiscounted_reward(h)` and `discounted_reward(h)` will return the total rewards collected over the trajectory. `n_steps(h)` returns the number of steps in the history. `exception(h)` and `backtrace(h)` can be used to hold an exception if the simulation failed to finish.

`view(h, range)` (e.g. `view(h, 1:n_steps(h)-4)`) can be used to create a view of the history object `h` that only contains a certain range of steps. The object returned by `view` is a `SimHistory` that can be iterated through and manipulated just like a complete `SimHistory`.

## Concrete Types

There are two concrete types of `SimHistory` depending on whether the problem was an MDP or a POMDP.

```@docs
MDPHistory
POMDPHistory
```
