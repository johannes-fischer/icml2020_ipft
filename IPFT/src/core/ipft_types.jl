"""Entropy MCTS solver with IPFT

Derived from MCTS-DPW (JuliaPOMDP/MCTS.jl/src/dpw_types.jl)

Fields:

    depth::Int64:
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64:
        Specified how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    k_action::Float64
    alpha_action::Float64
    k_observation::Float64
    alpha_observation::Float64
        These constants control the double progressive widening. A new state
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k:10, alpha:0.5

    lambda::Float64
        Hyper parameter balancing reward and information reward

    keep_tree::Bool
        If true, store the tree in the planner for reuse at the next timestep (and every time it is used in the future). There is a computational cost for maintaining the state dictionary necessary for this.
        default: false

    reuse_states::Bool
        If true, reuses first sample in a tree node instead of sampling new particles in every visit

    enable_action_pw::Bool
        If true, enable progressive widening on the action space; if false just use the whole action space.
        default: true

    check_repeat_state::Bool
    check_repeat_action::Bool
        When constructing the tree, check whether a state or action has been seen before (there is a computational cost to maintaining the dictionaries necessary for this)
        default: true

    tree_in_info::Bool:
        If true, return the tree in the info dict when action_info is called. False by default because it can use a lot of memory if histories are being saved.
        default: false

    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number.
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will always be set to that number.
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will always be set to that number.
        default: 0

    next_action::Any
        Function or object used to choose the next action to be considered for progressive widening.
        The next action is determined based on the MDP, the state, `s`, and the current `IPFTStateNode`, `snode`.
        If this is a function `f`, `f(mdp, s, snode)` will be called to set the value.
        If this is an object `o`, `next_action(o, mdp, s, snode)` will be called.
        default: RandomActionGenerator(rng)

    default_action::Any
        Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
        If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
        If this is a Policy `p`, `action(p, belief)` will be called.
        If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.
        default: `ExceptionRethrow()`
"""
mutable struct IPFTSolver <: AbstractMCTSSolver
    depth::Int
    criterion::AbstractMultiObjectiveCriterion
    final_criterion::AbstractMultiObjectiveCriterion
    # exploration_constant::Float64
    n_iterations::Int
    max_time::Float64
    k_action::Float64
    alpha_action::Float64
    k_observation::Float64
    alpha_observation::Float64
    keep_tree::Bool
    reuse_states::Bool
    enable_action_pw::Bool
    check_repeat_state::Bool
    check_repeat_action::Bool
    tree_in_info::Bool
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    next_action::Any
    default_action::Any
end

"""
    IPFTSolver()

Use keyword arguments to specify values for the fields
"""
function IPFTSolver(;depth::Int=10,
                    criterion::AbstractMultiObjectiveCriterion=MultiObjectiveUCB([1.0,0.0],1.0),
                    final_criterion::AbstractMultiObjectiveCriterion=MultiObjective(criterion),
                    # exploration_constant::Float64=1.0,
                    n_iterations::Int=100,
                    max_time::Float64=Inf,
                    k_action::Float64=10.0,
                    alpha_action::Float64=0.5,
                    k_observation::Float64=10.0,
                    alpha_observation::Float64=0.5,
                    keep_tree::Bool=false,
                    reuse_states::Bool=false,
                    enable_action_pw::Bool=true,
                    check_repeat_state::Bool=true,
                    check_repeat_action::Bool=true,
                    tree_in_info::Bool=false,
                    rng::AbstractRNG=Random.GLOBAL_RNG,
                    estimate_value::Any = RolloutEstimator(RandomSolver(rng)),
                    # init_Q::Any = 0.0,
                    init_Q::Any = [0.0, 0.0],
                    init_N::Any = 0,
                    next_action::Any = RandomActionGenerator(rng),
                    default_action::Any = ExceptionRethrow()
                   )
    IPFTSolver(depth, criterion, final_criterion, n_iterations, max_time, k_action, alpha_action, k_observation, alpha_observation, keep_tree, reuse_states, enable_action_pw, check_repeat_state, check_repeat_action, tree_in_info, rng, estimate_value, init_Q, init_N, next_action, default_action)
end

mutable struct IPFTree{S,A}
    # for each state node
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    s_labels::Vector{S}
    s_lookup::Dict{S, Int}

    # for each state-action node
    n::Vector{Int}
    q::Vector{Vector{Float64}}
    transitions::Vector{Vector{Tuple{Int,S,Array{Float64,1}}}}
    a_labels::Vector{A}
    a_lookup::Dict{Tuple{Int,A}, Int}

    # for tracking transitions
    n_a_children::Vector{Int}
    unique_transitions::Set{Tuple{Int,Int}}


    function IPFTree{S,A}(sz::Int=1000) where {S,A}
        sz = min(sz, 100_000)
        return new(sizehint!(Int[], sz),
                   sizehint!(Vector{Int}[], sz),
                   sizehint!(S[], sz),
                   Dict{S, Int}(),

                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(Vector{Tuple{Int,Float64}}[], sz),
                   sizehint!(A[], sz),
                   Dict{Tuple{Int,A}, Int}(),

                   sizehint!(Int[], sz),
                   Set{Tuple{Int,Int}}()
                  )
    end
end

function insert_state_node!(tree::IPFTree{S,A}, s::S, maintain_s_lookup=true) where {S,A}
    push!(tree.total_n, 0)
    push!(tree.children, Int[])
    push!(tree.s_labels, s)
    snode = length(tree.total_n)
    if maintain_s_lookup
        tree.s_lookup[s] = snode
    end
    return snode
end

function insert_action_node!(tree::IPFTree{S,A}, snode::Int, a::A, n0::Int, q0::Vector{Float64}, maintain_a_lookup=true) where {S,A}
    push!(tree.n, n0)
    push!(tree.q, q0)
    push!(tree.a_labels, a)
    push!(tree.transitions, Vector{Tuple{Int,Float64}}[])
    sanode = length(tree.n)
    push!(tree.children[snode], sanode)
    push!(tree.n_a_children, 0)
    if maintain_a_lookup
        tree.a_lookup[(snode, a)] = sanode
    end
    return sanode
end

Base.isempty(tree::IPFTree) = isempty(tree.n) && isempty(tree.q)

struct IPFTStateNode{S,A} <: AbstractStateNode
    tree::IPFTree{S,A}
    index::Int
end

total_n(n::IPFTStateNode) = n.tree.total_n[n.index]
children(n::IPFTStateNode) = n.tree.children[n.index]
n_children(n::IPFTStateNode) = length(children(n))
isroot(n::IPFTStateNode) = n.index == 1

mutable struct IPFTPlanner{P<:Union{MDP,POMDP}, S, A, SE, NA, RNG} <: InformationMCTSPlanner{P}
    solver::IPFTSolver
    mdp::P
    tree::Union{Nothing, IPFTree{S,A}}
    solved_estimate::SE
    next_action::NA
    rng::RNG
end

function IPFTPlanner(solver::IPFTSolver, mdp::P) where P<:Union{MDP,POMDP}
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return IPFTPlanner{P,
                      statetype(P),
                      actiontype(P),
                      typeof(se),
                      typeof(solver.next_action),
                      typeof(solver.rng)}(solver,
                                            mdp,
                                            nothing,
                                            se,
                                            solver.next_action,
                                            solver.rng
    )
end

Random.seed!(p::IPFTPlanner, seed) = Random.seed!(p.rng, seed)
