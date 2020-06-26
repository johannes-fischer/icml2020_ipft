mutable struct LocalApproximationValueIterationSolver{I<:LocalFunctionApproximator, RNG<:AbstractRNG} <: Solver
    interp::I # Will be copied over by value to each policy
    max_iterations::Int64 # max number of iterations
    belres::Float64 # the Bellman Residual
    verbose::Bool # Whether to print while solving or not
    rng::RNG # Seed if req'd
    is_mdp_generative::Bool # Whether to treat underlying MDP model as generative
    n_generative_samples::Int64 # If underlying model generative, how many samples to use
end

# Default constructor
function LocalApproximationValueIterationSolver(interp::I;
                                                max_iterations::Int64=100, belres::Float64=1e-3,
                                                verbose::Bool=false, rng::RNG=Random.GLOBAL_RNG,
                                                is_mdp_generative::Bool=false, n_generative_samples::Int64=0) where {I<:LocalFunctionApproximator, RNG<:AbstractRNG}
    return LocalApproximationValueIterationSolver(interp,max_iterations, belres, verbose, rng, is_mdp_generative, n_generative_samples)
end

# Unparameterized constructor just for getting requirements
function LocalApproximationValueIterationSolver()
    throw(ArgumentError("LocalApproximationValueIterationSolver needs a LocalFunctionApproximator object for construction!"))
end


# NOTE : We work directly with the value function
# And extract actions at the end by using the interpolation object
mutable struct LocalApproximationValueIterationPolicy{I<:LocalFunctionApproximator, RNG<:AbstractRNG} <: Policy
    interp::I # General approximator to be used in VI
    action_map::Vector # Maps the action index to the concrete action type
    mdp::Union{MDP,POMDP} # Uses the model for indexing in the action function
    is_mdp_generative::Bool # (Copied from solver.is_mdp_generative)
    n_generative_samples::Int64 # (Copied from solver.n_generative_samples)
    rng::RNG # (Copied from solver.rng)
end

# The policy can be created using the MDP and solver information
# The policy's function approximation object (interp) is obtained by deep-copying over the
# solver's interp object. The other policy parameters are also obtained from the solver
function LocalApproximationValueIterationPolicy(mdp::Union{MDP,POMDP},
                                                solver::LocalApproximationValueIterationSolver)
    return LocalApproximationValueIterationPolicy(deepcopy(solver.interp),ordered_actions(mdp),mdp,
                                                  solver.is_mdp_generative,solver.n_generative_samples,solver.rng)
end


@POMDP_require solve(solver::LocalApproximationValueIterationSolver, mdp::Union{MDP,POMDP}) begin

    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req n_actions(::P)
    @subreq ordered_actions(mdp)

    @req actionindex(::P, ::A)
    @req actions(::P, ::S)
    as = actions(mdp)
    a = first(as)

    @req convert_s(::Type{S},::V where V <: AbstractVector{Float64},::P)
    @req convert_s(::Type{V} where V <: AbstractVector{Float64},::S,::P)

    # Have different requirements depending on whether solver MDP is generative or explicit
    if solver.is_mdp_generative
        @req generate_sr(::P, ::S, ::A, ::typeof(solver.rng))
    else
        @req transition(::P, ::S, ::A)
        pts = get_all_interpolating_points(solver.interp)
        pt = first(pts)
        ss = POMDPs.convert_s(S,pt,mdp)
        dist = transition(mdp, ss, a)
        D = typeof(dist)
        @req support(::D)
    end

end


function solve(solver::LocalApproximationValueIterationSolver, mdp::Union{MDP,POMDP})

    @warn_requirements solve(solver,mdp)

    # Ensure that generative model has a non-zero number of samples
    if solver.is_mdp_generative
        @assert solver.n_generative_samples > 0
    end

    # Solver parameters
    max_iterations = solver.max_iterations
    belres = solver.belres
    discount_factor = discount(mdp)

    # Initialize the policy
    policy = LocalApproximationValueIterationPolicy(mdp,solver)

    total_time::Float64 = 0.0
    iter_time::Float64 = 0.0

    # Get attributes of interpolator
    # Since the policy object is created by the solver, it directly
    # modifies the value of the interpolator of the created policy
    num_interps::Int = n_interpolating_points(policy.interp)
    interp_points::Vector = get_all_interpolating_points(policy.interp)
    interp_values::Vector = get_all_interpolating_values(policy.interp)

    # Obtain the vector of states by converting the corresponding
    # vector of interpolation points/samples to the state type
    # using the user-provided convert_s function
    S = statetype(typeof(mdp))
    interp_states = Vector{S}(undef, num_interps)
    for (i,pt) in enumerate(interp_points)
        interp_states[i] = POMDPs.convert_s(S, pt, mdp)
    end

    # Outer loop for Value Iteration
    for i = 1 : max_iterations
        residual::Float64 = 0.0
        iter_time = @elapsed begin

        for (istate,s) in enumerate(interp_states)
            sub_aspace = actions(mdp,s)

            if isterminal(mdp, s)
                interp_values[istate] = 0.0
            else
                old_util = interp_values[istate]
                max_util = -Inf

                for a in sub_aspace
                    iaction = actionindex(mdp,a)
                    u::Float64 = 0.0

                    # Do bellman backup based on generative / explicit model
                    if solver.is_mdp_generative
                        # Generative Model
                        for j in 1:solver.n_generative_samples
                            sp, r = generate_sr(mdp, s, a, solver.rng)
                            u += r

                            # Only interpolate sp if it is non-terminal
                            # ADD A TO ENABLE TERMINAL ACTIONS
                            if !isterminal(mdp,sp,a)
                                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                                u += discount_factor*compute_value(policy.interp, sp_point)
                            end
                        end
                        u = u / solver.n_generative_samples
                    else
                        # Explicit Model
                        dist = transition(mdp,s,a)
                        for (sp, p) in weighted_iterator(dist)
                            p == 0.0 ? continue : nothing
                            r = reward(mdp, s, a, sp)
                            u += p*r

                            # Only interpolate sp if it is non-terminal
                            # ADD A TO ENABLE TERMINAL ACTIONS
                            if !isterminal(mdp,sp,a)
                                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                                u += p * (discount_factor*compute_value(policy.interp, sp_point))
                            end
                        end # next-states
                    end

                    max_util = (u > max_util) ? u : max_util
                end #action

                # Update this interpolant value
                interp_values[istate] = max_util
                util_diff = abs(max_util - old_util)
                util_diff > residual ? (residual = util_diff) : nothing
            end
        end #state

        end #time
        total_time += iter_time
        solver.verbose ? @printf("[Iteration %-4d] residual: %10.3G | iteration runtime: %10.3f ms, (%10.3G s total)\n", i, residual, iter_time*1000.0, total_time) : nothing
        residual < belres ? break : nothing

    end #main
    return policy
end


function value(policy::LocalApproximationValueIterationPolicy, s::S) where S

    # Call the conversion function on the state to get the corresponding vector
    # That represents the point at which to interpolate the function
    s_point = POMDPs.convert_s(Vector{Float64}, s, policy.mdp)
    val = compute_value(policy.interp, s_point)
    return val
end

# Not explicitly stored in policy - extract from value function interpolation
function action(policy::LocalApproximationValueIterationPolicy, s::S) where S

    mdp = policy.mdp
    best_a_idx = -1
    max_util = -Inf
    sub_aspace = actions(mdp,s)
    discount_factor = discount(mdp)


    for a in iterator(sub_aspace)

        iaction = actionindex(mdp, a)
        u::Float64 = action_value(policy,s,a)

        if u > max_util
            max_util = u
            best_a_idx = iaction
        end
    end

    return policy.action_map[best_a_idx]
end

# Compute the action-value for some state-action pair
# This is also used in the above function
function action_value(policy::LocalApproximationValueIterationPolicy, s::S, a::A) where {S,A}

    mdp = policy.mdp
    discount_factor = discount(mdp)

    u::Float64 = 0.0

    # As in solve(), do different things based on whether
    # mdp is generative or explicit
    if policy.is_mdp_generative
        # BEFORE:
        # for j in 1:policy.n_generative_samples
        #     sp, r = generate_sr(mdp, s, a, policy.rng)
        #     sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
        #     u += r + discount_factor*compute_value(policy.interp, sp_point)
        # end
        # AFTER ADDING TERMINAL ACTIONS:
        for j in 1:solver.n_generative_samples
            sp, r = generate_sr(mdp, s, a, solver.rng)
            u += r

            if !isterminal(mdp,sp,a)
                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                u += discount_factor*compute_value(policy.interp, sp_point)
            end
        end
        u = u / policy.n_generative_samples
    else
        dist = transition(mdp,s,a)
        for (sp, p) in weighted_iterator(dist)
            p == 0.0 ? continue : nothing
            r = reward(mdp, s, a, sp)
            u += p*r

            # Only interpolate sp if it is non-terminal
            # ADD A TO ENABLE TERMINAL ACTIONS
            if !isterminal(mdp,sp,a)
                sp_point = POMDPs.convert_s(Vector{Float64}, sp, mdp)
                u += p*(discount_factor*compute_value(policy.interp, sp_point))
            end
        end
    end

    return u
end
