"""
Represents everything needed to run and record a single simulation, including model, initial conditions, and metadata.

A vector of `Sim` objects can be executed with [`run`](@ref) or [`run_parallel`](@ref).

## Keyword Arguments
- `rng::AbstractRNG=Random.GLOBAL_RNG`
- `max_steps::Int=typemax(Int)`
- `simulator::Simulator=HistoryRecorder(rng=rng, max_steps=max_steps)`
- `metadata::NamedTuple a named tuple (or dictionary) of metadata for the sim that will be recorded, e.g. `(solver_iterations=500,)`.
"""
abstract type Sim end

struct POMDPSim <: Sim
    simulator::Simulator
    pomdp::POMDP
    policy::Policy
    updater::Updater
    initial_belief::Any
    initialstate::Any
    metadata::NamedTuple
end

problem(sim::POMDPSim) = sim.pomdp

struct MDPSim <: Sim
    simulator::Simulator
    mdp::MDP
    policy::Policy
    initialstate::Any
    metadata::NamedTuple
end

problem(sim::MDPSim) = sim.mdp

"""
    Sim(p::POMDP, policy::Policy, metadata=(note="a note",))
    Sim(p::POMDP, policy::Policy[, updater[, initial_belief[, initialstate]]]; kwargs...)

Create a `Sim` object that represents a POMDP simulation.
"""
function Sim(pomdp::POMDP,
             policy::Policy,
             up=updater(policy),
             initial_belief=initialstate_distribution(pomdp),
             initialstate=nothing;
             rng::AbstractRNG=Random.GLOBAL_RNG,
             max_steps::Int=typemax(Int),
             simulator::Simulator=HistoryRecorder(rng=rng, max_steps=max_steps),
             metadata = NamedTuple()
            )

    if initialstate == nothing && statetype(pomdp) != Nothing
        is = rand(rng, initial_belief)
    else
        is = initialstate
    end
    return POMDPSim(simulator, pomdp, policy, up, initial_belief, is, merge(NamedTuple(), metadata))
end

"""
    Sim(p::MDP, policy::Policy, metadata=(note="a note",))
    Sim(p::MDP, policy::Policy[, initialstate]; kwargs...)

Create a `Sim` object that represents a MDP simulation.
"""
function Sim(mdp::MDP,
             policy::Policy,
             initialstate=nothing;
             rng::AbstractRNG=Random.GLOBAL_RNG,
             max_steps::Int=typemax(Int),
             simulator::Simulator=HistoryRecorder(rng=rng, max_steps=max_steps),
             metadata = NamedTuple()
            )

    if initialstate == nothing && statetype(mdp) != Nothing
        is = POMDPs.initialstate(mdp, rng) 
    else
        is = initialstate
    end
    return MDPSim(simulator, mdp, policy, is, merge(NamedTuple(), metadata))
end

POMDPs.simulate(s::POMDPSim) = simulate(s.simulator, s.pomdp, s.policy, s.updater, s.initial_belief, s.initialstate)
POMDPs.simulate(s::MDPSim) = simulate(s.simulator, s.mdp, s.policy, s.initialstate)

default_process(s::Sim, r::Float64) = (reward=r,)
default_process(s::Sim, hist::SimHistory) = default_process(s, discounted_reward(hist))

run_parallel(queue::AbstractVector; kwargs...) = run_parallel(default_process, queue; kwargs...)

"""
    run_parallel(queue::Vector{Sim})
    run_parallel(f::Function, queue::Vector{Sim})

Run `Sim` objects in `queue` in parallel and return results as a `DataFrame`.

By default, the `DataFrame` will contain the reward for each simulation and the metadata provided to the sim.

# Arguments
- `queue`: List of `Sim` objects to be executed
- `f`: Function to process the results of each simulation
This function should take two arguments, (1) the `Sim` that was executed and (2) the result of the simulation, by default a `SimHistory`. It should return a named tuple that will appear in the dataframe. See Examples below.

## Keyword Arguments
- `progress`: a `ProgressMeter.Progress` for showing progress through the simulations; `progress=false` will suppress the progress meter

# Examples

```julia
run_parallel(queue) do sim, hist
    return (n_steps=n_steps(hist), reward=discounted_reward(hist))
end
```
will return a dataframe with with the number of steps and the reward in it.
"""
function run_parallel(process::Function, queue::AbstractVector;
                      progress=Progress(length(queue), desc="Simulating..."),
                      proc_warn=true)

    #=
    frame_lines = pmap(progress, queue) do sim
        result = simulate(sim)
        return process(sim, result)
    end
    =#

    np = nprocs()
    if np == 1 && proc_warn
        @warn("""
             run_parallel(...) was started with only 1 process, so simulations will be run in serial. 

             To supress this warning, use run_parallel(..., proc_warn=false).

             To use multiple processes, use addprocs() or the -p option (e.g. julia -p 4).
             """)
    end
    n = length(queue)
    i = 1
    prog = 0
    # based on the simple implementation of pmap here: https://docs.julialang.org/en/latest/manual/parallel-computing
    frame_lines = Vector{Any}(missing, n)
    nextidx() = (idx=i; i+=1; idx)
    prog_lock = ReentrantLock()
    @sync begin 
        for p in 1:np
            if np == 1 || p != myid()
                @async begin
                    while true
                        idx = nextidx()
                        if idx > n
                            break
                        end
                        frame_lines[idx] = remotecall_fetch(p, queue[idx]) do sim
                            result = simulate(sim)
                            output = process(sim, result)
                            return merge(sim.metadata, output)
                        end
                        if progress isa Progress
                            lock(prog_lock)
                            update!(progress, prog+=1)
                            unlock(prog_lock)
                        end
                    end
                end
            end
        end
    end
    if progress isa Progress
        lock(prog_lock)
        finish!(progress)
        unlock(prog_lock)
    end

    return create_dataframe(frame_lines)
end

Base.run(queue::AbstractVector) = run(default_process, queue)

"""
    run(queue::Vector{Sim})
    run(f::Function, queue::Vector{Sim})

Run the `Sim` objects in `queue` on a single process and return the results as a dataframe.

See `run_parallel` for more information.
"""
function Base.run(process::Function, queue::AbstractVector; show_progress=true)
    lines = []
    if show_progress
        @showprogress for sim in queue
            result = simulate(sim)
            output = process(sim, result)
            line = merge(sim.metadata, output)
            push!(lines, line)
        end
    else
        for sim in queue
            result = simulate(sim)
            output = process(sim, result)
            line = merge(sim.metadata, output)
            push!(lines, line)
        end
    end
    return create_dataframe(lines)
end

function create_dataframe(lines::Vector)
    master = Dict{Symbol, AbstractVector}()
    for line in lines
        push_line!(master, line)
    end
    return DataFrame(master)
end

function push_line!(d::Dict{Symbol, AbstractVector}, line::NamedTuple)
    if isempty(d)
        len = 0
    else
        len = length(first(values(d)))
    end
    for (key, val) in pairs(line)
        T = typeof(val)
        if !haskey(d, key)
            d[key] = Vector{Union{T,Missing}}(missing, len)
        end
        data = d[key]
        if !isa(val,eltype(data))
            d[key] = convert(Array{promote_type(typeof(val), eltype(data)),1}, data)
        end
        push!(d[key], val)
    end
    for da in values(d)
        if length(da) < len + 1
            push!(da, missing)
        end
    end
    return d
end
