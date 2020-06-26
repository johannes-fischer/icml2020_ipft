[![Build Status](https://travis-ci.org/JuliaPOMDP/LocalApproximationValueIteration.jl.svg?branch=master)](https://travis-ci.org/JuliaPOMDP/LocalApproximationValueIteration.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaPOMDP/LocalApproximationValueIteration.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaPOMDP/LocalApproximationValueIteration.jl?branch=master)

# LocalApproximationValueIteration

This package implements the Local Approximation Value Iteration algorithm in Julia for solving
Markov Decision Processes (MDPs). Algorithmically it is very similar to the [DiscreteValueIteration.jl](https://github.com/JuliaPOMDP/DiscreteValueIteration.jl) 
package, but it represents the state space in a fundamentally different manner, as explained below.
As with `DiscreteValueIteration`, the user should define the problem according to the API in
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). Examples of problem definitions can be found in
[POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl).

## Installation

Start Julia and run the following:

```julia
using POMDPs
POMDPs.add("LocalApproximationValueIteration")
```

## How it Works

This solver is one example of _Approximate Dynamic Programming_, which tries to find approximately optimal
value functions and policies for large or continuous state spaces. The approach of Local Approximation Value
Iteration assumes that states that are _near_ each other (by some appropriate distance metric) will have similar
values, and it computes the values at some (user-defined) finite set of states, and interpolates the value
function over the entire state space using some local function approximation technique. Details of this approach
are described in **Section 4.5.1** of the book [Decision Making Under Uncertainty : Theory and Application](https://dl.acm.org/citation.cfm?id=2815660).

## State Space Representation

For value function approximation, the solver depends on the [LocalFunctionApproximation.jl](https://github.com/sisl/LocalFunctionApproximation.jl)
package. The `LocalApproximationValueIteration` solver must be
initialized with an appropriate `LocalFunctionApproximator` object that approximates
the computed value function over the entire state space by either interpolation over a multi-dimensional grid discretization
of the state space, or by k-nearest-neighbor averaging
with a randomly drawn set of state space samples. The resulting policy uses this object to compute the action-value
function or the best action for any arbitrary state query.

A key operational requirement that the solver has from the MDP is that any state can be represented via an equivalent
real-valued vector. This is enforced by the two `convert_s` function requirements that convert an instance of
the MDP State type to a real-valued vector and vice versa. The signatures for these methods are:

```julia
convert_s(::Type{S},::V where V <: AbstractVector{Float64},::P)
convert_s(::Type{V} where V <: AbstractVector{Float64},::S,::P)
```

The user is required to implement the above two functions for the `State` type of their MDP problem model. An example of this
is shown in `test/runtests_versus_discrete_vi.jl` for the [GridWorld](https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/gridworld.jl) model:

```julia
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::GridWorldState, mdp::GridWorld)
    v = SVector{3,Float64}(s.x, s.y, convert(Float64,s.done))
    return v
end

function POMDPs.convert_s(::Type{GridWorldState}, v::AbstractVector{Float64}, mdp::GridWorld)
    s = GridWorldState(round(Int64, v[1]), round(Int64, v[2]), convert(Bool, v[3]))
end
```

## Usage

`POMDPs.jl` has a macro `@requirements_info` that determines the functions necessary to use some solver on some specific MDP model. As mentioned above, the
`LocalApproximationValueIteration` solver depends on a `LocalFunctionApproximator` object and so that object must first be created to invoke
the requirements of the solver accordingly (check [here](http://juliapomdp.github.io/POMDPs.jl/latest/requirements) for more information). From our running example in `test/runtests_versus_discrete_vi.jl`, a function approximation object that uses grid interpolation 
(`LocalGIFunctionApproximator`) is created, after the appropriate `RectangleGrid` is 
constructed (Look at [GridInterpolations.jl](https://github.com/sisl/GridInterpolations.jl/blob/master/src/GridInterpolations.jl/) for more details about this).

```julia
using POMDPs, POMDPModels
using GridInterpolations
using LocalFunctionApproximation
using LocalApproximationValueIteration

VERTICES_PER_AXIS = 10 # Controls the resolutions along the grid axis
grid = RectangleGrid(linspace(1,100,VERTICES_PER_AXIS), linspace(1,100,VERTICES_PER_AXIS), [0.0, 1.0]) # Create the interpolating grid
interp = LocalGIFunctionApproximator(grid)  # Create the local function approximator using the grid

@requirements_info LocalApproximationValueIterationSolver(interp) GridWorld() # Check if the solver requirements are met
```

The user should modify the above steps depending on the kind of interpolation and the necessary parameters they want. We have delegated this step to the user
as it is extremely problem and domain specific. Note that the solver supports both explicit and generative transition models for the MDP (more on that [here](http://juliapomdp.github.io/POMDPs.jl/latest/def_pomdp)).
The `.is_mdp_generative` and `.n_generative_samples` arguments of the `LocalApproximationValueIteration` solver should be set accordingly, and there are different
`@requirements` depending on which kind of model the MDP has.

Once all the necessary functions have been defined, the solver can be created.  A `GridWorld` MDP is defined with grid size 100 x 100 and appropriate reward states:

```julia
mdp = GridWorld(sx=100, sy=100, rs=rstates, rv=rvect)
```

Finally, the solver can be created using the function approximation object and other necessary parameters
(this model is explicit), and the MDP can be solved:

```julia
approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations=1000, is_mdp_generative=false)
approx_policy = solve(approx_solver, mdp)
```

The API for querying the final policy object is identical to `DiscreteValueIteration`, i.e. the `action` and `value` functions can be called for the solved MDP:

```julia
v = value(policy, s)  # returns the approximately optimal value for state s
a = action(policy, s) # returns the approximately optimal action for state s
```
