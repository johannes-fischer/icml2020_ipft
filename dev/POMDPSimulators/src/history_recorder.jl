# HistoryRecorder
# maintained by @zsunberg

"""
A simulator that records the history for later examination

The simulation will be terminated when either
1) a terminal state is reached (as determined by `isterminal()` or
2) the discount factor is as small as `eps` or
3) max_steps have been executed

Keyword Arguments:
    - `rng`: The random number generator for the simulation
    - `capture_exception::Bool`: whether to capture an exception and store it in the history, or let it go uncaught, potentially killing the script
    - `show_progress::Bool`: show a progress bar for the simulation
    - `eps`
    - `max_steps`
    - `sizehint::Int`: the expected length of the simulation (for preallocation)

Usage (optional arguments in brackets):

    hr = HistoryRecorder()
    history = simulate(hr, pomdp, policy, [updater [, init_belief [, init_state]]])
"""
mutable struct HistoryRecorder <: Simulator
    rng::AbstractRNG

    # options
    capture_exception::Bool
    show_progress::Bool

    # optional: if these are null, they will be ignored
    max_steps::Union{Nothing,Any}
    eps::Union{Nothing,Any}
    sizehint::Union{Nothing,Integer}
end

# This is the only stable constructor
function HistoryRecorder(;rng=MersenneTwister(rand(UInt32)),
                          eps=nothing,
                          max_steps=nothing,
                          sizehint=nothing,
                          capture_exception=false,
                          show_progress=false)
    return HistoryRecorder(rng, capture_exception, show_progress, max_steps, eps, sizehint)
end

@POMDP_require simulate(sim::HistoryRecorder, pomdp::POMDP, policy::Policy) begin
    @req updater(::typeof(policy))
    up = updater(policy)
    @subreq simulate(sim, pomdp, policy, up)
end

@POMDP_require simulate(sim::HistoryRecorder, pomdp::POMDP, policy::Policy, bu::Updater) begin
    @req initialstate_distribution(::typeof(pomdp))
    dist = initialstate_distribution(pomdp)
    @subreq simulate(sim, pomdp, policy, bu, dist)
end

function simulate(sim::HistoryRecorder, pomdp::POMDP, policy::Policy, bu::Updater=updater(policy))
    dist = initialstate_distribution(pomdp)
    return simulate(sim, pomdp, policy, bu, dist)
end

@POMDP_require simulate(sim::HistoryRecorder, pomdp::POMDP, policy::Policy, bu::Updater, dist::Any) begin
    P = typeof(pomdp)
    S = statetype(pomdp)
    A = actiontype(pomdp)
    O = obstype(pomdp)
    @req initialize_belief(::typeof(bu), ::typeof(dist))
    @req isterminal(::P, ::S)
    @req discount(::P)
    @req generate_sor(::P, ::S, ::A, ::typeof(sim.rng))
    b = initialize_belief(bu, dist)
    B = typeof(b)
    @req action(::typeof(policy), ::B)
    @req update(::typeof(bu), ::B, ::A, ::O)
end

function simulate(sim::HistoryRecorder,
                           pomdp::POMDP{S,A,O},
                           policy::Policy,
                           bu::Updater,
                           initialstate_dist::Any,
                           initialstate::Any=get_initialstate(sim, initialstate_dist)
                  ) where {S,A,O}

    initial_belief = initialize_belief(bu, initialstate_dist)
    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end
    if sim.eps != nothing
        max_steps = min(max_steps, ceil(Int,log(sim.eps)/log(discount(pomdp))))
    end
    if sim.sizehint == nothing
        sizehint = min(max_steps, 1000)
    else
        sizehint = sim.sizehint
    end

    # aliases for the histories to make the code more concise
    sh = sizehint!(Vector{S}(undef, 0), sizehint)
    ah = sizehint!(Vector{A}(undef, 0), sizehint)
    oh = sizehint!(Vector{O}(undef, 0), sizehint)
    bh = sizehint!(Vector{typeof(initial_belief)}(undef, 0), sizehint)
    rh = sizehint!(Vector{Float64}(undef, 0), sizehint)
    ih = sizehint!(Vector{Any}(undef, 0), sizehint)
    aih = sizehint!(Vector{Any}(undef, 0), sizehint)
    uih = sizehint!(Vector{Any}(undef, 0), sizehint)
    exception = nothing
    backtrace = nothing

    push!(sh, initialstate)
    push!(bh, initial_belief)

    if sim.show_progress
        if (sim.max_steps == nothing) && (sim.eps == nothing)
            error("If show_progress=true in a HistoryRecorder, you must also specify max_steps or eps.")
        end
        prog = Progress(max_steps, "Simulating..." )
    end

    disc = 1.0
    step = 1

    try
        while !isterminal(pomdp, sh[step], get(ah,step-1,nothing), get(oh,step-1,nothing)) && step <= max_steps
            a, ai = action_info(policy, bh[step])
            push!(ah, a)
            push!(aih, ai)

            sp, o, r, i = generate_sori(pomdp, sh[step], ah[step], sim.rng)

            push!(sh, sp)
            push!(oh, o)
            push!(rh, r)
            push!(ih, i)

            bp, ui = update_info(bu, bh[step], ah[step], oh[step])
            bh = push_belief(bh, bp)
            push!(uih, ui)

            step += 1

            if sim.show_progress
                next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            exception = ex
            backtrace = catch_backtrace()
        else
            rethrow(ex)
        end
    end

    if sim.show_progress
        finish!(prog)
    end

    return POMDPHistory(sh, ah, oh, bh, rh, ih, aih, uih, discount(pomdp), exception, backtrace)
end

@POMDP_require simulate(sim::HistoryRecorder, mdp::MDP, policy::Policy) begin
    init_state = initialstate(mdp, sim.rng)
    @subreq simulate(sim, mdp, policy, init_state)
end

@POMDP_require simulate(sim::HistoryRecorder, mdp::MDP, policy::Policy, initialstate::Any) begin
    P = typeof(mdp)
    S = statetype(mdp)
    A = actiontype(mdp)
    @req isterminal(::P, ::S)
    @req action(::typeof(policy), ::S)
    @req generate_sr(::P, ::S, ::A, ::typeof(sim.rng))
    @req discount(::P)
end

function simulate(sim::HistoryRecorder,
                  mdp::MDP{S,A}, policy::Policy,
                  init_state::S=get_initialstate(sim, mdp)) where {S,A}

    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end
    if sim.eps != nothing
        max_steps = min(max_steps, ceil(Int,log(sim.eps)/log(discount(mdp))))
    end
    if sim.sizehint == nothing
        sizehint = min(max_steps, 1000)
    else
        sizehint = sim.sizehint
    end

    # aliases for the histories to make the code more concise
    sh = sizehint!(Vector{S}(undef, 0), sizehint)
    ah = sizehint!(Vector{A}(undef, 0), sizehint)
    rh = sizehint!(Vector{Float64}(undef, 0), sizehint)
    ih = sizehint!(Vector{Any}(undef, 0), sizehint)
    aih = sizehint!(Vector{Any}(undef, 0), sizehint)
    exception = nothing
    backtrace = nothing

    if sim.show_progress
        prog = Progress(max_steps, "Simulating..." )
    end

    push!(sh, init_state)

    disc = 1.0
    step = 1

    try
        while !isterminal(mdp, sh[step], get(ah,step-1,nothing)) && step <= max_steps
            a, ai = action_info(policy, sh[step])
            push!(ah, a)
            push!(aih, ai)

            sp, r, i = generate_sri(mdp, sh[step], ah[step], sim.rng)

            push!(sh, sp)
            push!(rh, r)
            push!(ih, i)

            disc *= discount(mdp)
            step += 1

            if sim.show_progress
                next!(prog)
            end
        end
    catch ex
        if sim.capture_exception
            exception = ex
            backtrace = catch_backtrace()
        else
            rethrow(ex)
        end
    end

    if sim.show_progress
        finish!(prog)
    end

    return MDPHistory(sh, ah, rh, ih, aih, discount(mdp), exception, backtrace)
end

function get_initialstate(sim::Simulator, initialstate_dist)
    return rand(sim.rng, initialstate_dist)
end

function get_initialstate(sim::Simulator, mdp::Union{MDP,POMDP})
    return initialstate(mdp, sim.rng)
end

# this is kind of a hack in cases where the belief isn't stable
push_belief(bh::Vector{T}, b::T) where T = push!(bh, b)
function push_belief(bh::Vector{T}, b::B) where {B, T}
    if !(T isa Union) # if T is not already a Union, try making a Union of the two types; don't jump straight to Any
        new = Vector{Union{T,B}}(undef, length(bh)+1)
    else
        new = Vector{promote_type(T, B)}(undef, length(bh)+1)
    end
    new[1:end-1] = bh
    new[end] = b
    return new
end
