POMDPs.solve(solver::IPFTSolver, mdp::Union{POMDP,MDP}) = IPFTPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::IPFTPlanner)
    p.tree = nothing
end

MCTS.init_Q(q::Array{Float64}, mdp::Union{MDP,POMDP}, s, a) = q

"""
Construct an IPFT tree and choose the best action.
"""
POMDPs.action(p::IPFTPlanner, s) = first(action_info(p, s))

"""
Construct an IPFT tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::IPFTPlanner, s; tree_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  IPFT cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(p.mdp)
        @assert s isa S "s:$(typeof(s)), S=$S"
        A = actiontype(p.mdp)
        if p.solver.keep_tree
            if p.tree == nothing
                tree = IPFTree{S,A}(p.solver.n_iterations)
                p.tree = tree
            else
                tree = p.tree
            end
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            tree = IPFTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        nquery = 0
        start_us = CPUtime_us()
        for j = 1:p.solver.n_iterations
            nquery += 1
            @debug "Start simulation $j"
            simulate(p, snode, s, p.solver.depth)
            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                break
            end
        end
        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode = select_best(p.solver.final_criterion, IPFTStateNode(tree,snode), p.rng)
        # XXX some publications say to choose action that has been visited the most
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of IPFT.
"""
function POMDPModelTools.simulate(p::IPFTPlanner{P, S, A, SE, NA, RNG}, snode::Int, s::S, d::Int) where {P, S, A, SE, NA, RNG}
    sol = p.solver
    @debug "Run simulate on depth $(sol.depth - d)"
    tree = p.tree
    # s = tree.s_labels[snode]
    if d == 0
        return [0.0, 0.0]
    end
    # action progressive widening
    if sol.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(p.next_action, p.mdp, s, IPFTStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, p.mdp, s, a)
                insert_action_node!(tree, snode, a, n0,
                                    init_Q(sol.init_Q, p.mdp, s, a),
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in actions(p.mdp, s)
            n0 = init_N(sol.init_N, p.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, p.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end
    sanode = select_best(sol.criterion, IPFTStateNode(tree,snode), p.rng)
    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    if tree.n_a_children[sanode] <= sol.k_observation*tree.n[sanode]^sol.alpha_observation
        sp, r = generate_sr(p.mdp, s, a, p.rng)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end

        push!(tree.transitions[sanode], (spnode, sp, r))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, sp, r = rand(p.rng, tree.transitions[sanode])
        if !sol.reuse_states
            sp, r = generate_sr(p.mdp, s, a, p.rng)
        end
    end

    if isterminal(p.mdp, sp, a)
        q = r
    elseif new_node
        q = r + discount(p.mdp)*estimate_value(p.solved_estimate, p.mdp, sp, d-1)
    else
        q = r + discount(p.mdp)*simulate(p, spnode, sp, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q
end
