# StepSimulator
# maintained by @zsunberg

mutable struct StepSimulator <: Simulator
    rng::AbstractRNG
    max_steps::Union{Nothing,Any}
    spec
end
function StepSimulator(spec; rng=Random.GLOBAL_RNG, max_steps=nothing)
    return StepSimulator(rng, max_steps, spec)
end

function simulate(sim::StepSimulator, mdp::MDP{S}, policy::Policy, init_state::S=get_initialstate(sim, mdp)) where {S}
    symtuple = convert_spec(sim.spec, MDP)
    if sim.max_steps == nothing
        max_steps = typemax(Int64)
    else
        max_steps = sim.max_steps
    end
    return MDPSimIterator(symtuple, mdp, policy, sim.rng, init_state, max_steps)
end

function simulate(sim::StepSimulator, pomdp::POMDP, policy::Policy, bu::Updater=updater(policy))
    dist = initialstate_distribution(pomdp)
    return simulate(sim, pomdp, policy, bu, dist)
end

function simulate(sim::StepSimulator, pomdp::POMDP, policy::Policy, bu::Updater, dist::Any, initialstate=nothing)
    if initialstate == nothing
        initialstate = get_initialstate(sim, dist)
    end
    initial_belief = initialize_belief(bu, dist)
    symtuple = convert_spec(sim.spec, POMDP)
    if sim.max_steps == nothing
        max_steps = typemax(Int64)
    else
        max_steps = sim.max_steps
    end
    return POMDPSimIterator(symtuple, pomdp, policy, bu, sim.rng, initial_belief, initialstate, max_steps)
end

struct MDPSimIterator{SPEC, M<:MDP, P<:Policy, RNG<:AbstractRNG, S}
    mdp::M
    policy::P
    rng::RNG
    init_state::S
    max_steps::Int
end

function MDPSimIterator(spec::Union{Tuple, Symbol}, mdp::MDP, policy::Policy, rng::AbstractRNG, init_state, max_steps::Int)
    return MDPSimIterator{spec, typeof(mdp), typeof(policy), typeof(rng), typeof(init_state)}(mdp, policy, rng, init_state, max_steps)
end

Base.IteratorSize(::Type{<:MDPSimIterator}) = Base.SizeUnknown()

function Base.iterate(it::MDPSimIterator, is::Tuple{Int, S, A}=(1, it.init_state, nothing)) where {S,A}
    if (is[1] == 1 ? isterminal(it.mdp, is[2]) : isterminal(it.mdp, is[2:3]...)) || is[1] > it.max_steps
        return nothing
    end
    t = is[1]
    s = is[2]
    a, ai = action_info(it.policy, s)
    sp, r, i = generate_sri(it.mdp, s, a, it.rng)
    return (out_tuple(it, (s, a, r, sp, t, i, ai)), (t+1, sp, a))
end

struct POMDPSimIterator{SPEC, M<:POMDP, P<:Policy, U<:Updater, RNG<:AbstractRNG, B, S}
    pomdp::M
    policy::P
    updater::U
    rng::RNG
    init_belief::B
    init_state::S
    max_steps::Int
end
function POMDPSimIterator(spec::Union{Tuple,Symbol}, pomdp::POMDP, policy::Policy, up::Updater, rng::AbstractRNG, init_belief, init_state, max_steps::Int)
    return POMDPSimIterator{spec,
                            typeof(pomdp),
                            typeof(policy),
                            typeof(up),
                            typeof(rng),
                            typeof(init_belief),
                            typeof(init_state)}(pomdp,
                                                policy,
                                                up,
                                                rng,
                                                init_belief,
                                                init_state,
                                                max_steps)
end

Base.IteratorSize(::Type{<:POMDPSimIterator}) = Base.SizeUnknown()

function Base.iterate(it::POMDPSimIterator, is::Tuple{Int,S,A,O,B} = (1, it.init_state, nothing, nothing, it.init_belief)) where {S,A,O,B}
    if (is[1] == 1 ? isterminal(it.pomdp, is[2]) : isterminal(it.pomdp, is[2:4]...)) || is[1] > it.max_steps
        return nothing
    end
    t = is[1]
    s = is[2]
    b = is[5]
    a, ai = action_info(it.policy, b)
    sp, o, r, i = generate_sori(it.pomdp, s, a, it.rng)
    bp, ui = update_info(it.updater, b, a, o)
    return (out_tuple(it, (s, a, r, sp, t, i, ai, b, o, bp, ui)), (t+1, sp, a, o, bp))
end

const sym_to_ind = Dict(sym=>i for (i, sym) in enumerate(COMPLETE_POMDP_STEP))

@generated function out_tuple(it::Union{MDPSimIterator, POMDPSimIterator}, all::Tuple)
    spec = it.parameters[1]
    if isa(spec, Tuple)
        calls = []
        for sym in spec
            push!(calls, :($sym = all[$(sym_to_ind[sym])]))
        end

        return quote
            return ($(calls...),)
        end
    else
        @assert isa(spec, Symbol) "Invalid specification: $spec is not a Symbol or Tuple."
        return quote
            return all[$(sym_to_ind[spec])]
        end
    end
end

convert_spec(spec, T::Type{POMDP}) = convert_spec(spec, Set(tuple(:sp, :bp, :s, :a, :r, :b, :o, :i, :ai, :ui, :t)))
convert_spec(spec, T::Type{MDP}) = convert_spec(spec, Set(tuple(:sp, :s, :a, :r, :i, :ai, :t)))

function convert_spec(spec, recognized::Set{Symbol})
    conv = convert_spec(spec)
    for s in (isa(conv, Tuple) ? conv : tuple(conv))
        if !(s in recognized)
            @warn("uncrecognized symbol $s in step iteration specification $spec.")
        end
    end
    return conv
end

function convert_spec(spec::String)
    syms = [Symbol(m.match) for m in eachmatch(r"(sp|bp|ai|ui|s|a|r|b|o|i|t)", spec)]
    if length(syms) == 0
        error("$spec does not contain any valid symbols for step iterator output. Valid symbols are sp, bp, ai, ui, s, a, r, b, o, i, t")
    end
    if length(syms) == 1
        return Symbol(first(syms))
    else
        return tuple(syms...)
    end
end

function convert_spec(spec::Tuple)
    for s in spec
        @assert isa(s, Symbol)
    end
    return spec
end

convert_spec(spec::Symbol) = spec

"""
    stepthrough(problem, policy, [spec])
    stepthrough(problem, policy, [spec], [rng=rng], [max_steps=max_steps])
    stepthrough(mdp::MDP, policy::Policy, [init_state], [spec]; [kwargs...])
    stepthrough(pomdp::POMDP, policy::Policy, [up::Updater, [initial_belief, [initial_state]]], [spec]; [kwargs...])

Create a simulation iterator. This is intended to be used with for loop syntax to output the results of each step *as the simulation is being run*.

Example:

    pomdp = BabyPOMDP()
    policy = RandomPolicy(pomdp)

    for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps=10)
        println("in state \$s")
        println("took action \$o")
        println("received observation \$o and reward \$r")
    end

The optional `spec` argument can be a string, tuple of symbols, or single symbol and follows the same pattern as [`eachstep`](@ref) called on a `SimHistory` object.

Under the hood, this function creates a `StepSimulator` with `spec` and returns a `[PO]MDPSimIterator` by calling simulate with all of the arguments except `spec`. All keyword arguments are passed to the `StepSimulator` constructor.
"""
function stepthrough end # for documentation

function stepthrough(mdp::MDP, policy::Policy, spec::Union{String, Tuple, Symbol}=COMPLETE_MDP_STEP; kwargs...)
    sim = StepSimulator(spec; kwargs...)
    return simulate(sim, mdp, policy)
end

function stepthrough(mdp::MDP{S},
                     policy::Policy,
                     init_state::S,
                     spec::Union{String, Tuple, Symbol}=COMPLETE_MDP_STEP;
                     kwargs...) where {S}
    sim = StepSimulator(spec; kwargs...)
    return simulate(sim, mdp, policy, init_state)
end

function stepthrough(pomdp::POMDP, policy::Policy, args...; kwargs...)
    spec_included=false
    if !isempty(args) && isa(last(args), Union{String, Tuple, Symbol})
        spec = last(args)
        spec_included = true
        if spec isa statetype(pomdp) && length(args) == 3
            error("Ambiguity between `initial_state` and `spec` arguments in stepthrough. Please explicitly specify the initial state and spec.")
        end
    else
        spec = COMPLETE_POMDP_STEP
    end
    sim = StepSimulator(spec; kwargs...)
    return simulate(sim, pomdp, policy, args[1:end-spec_included]...)
end
