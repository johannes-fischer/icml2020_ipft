#=
Construct a full grid of 100 x 100 and a discrete grid of 10 x 10
Put reward states of all the same value from (40,40) to (60,60)
Run discrete VI and localapproxVI on each grid
Then randomly sample several points on full grid and compare 
=#

# State conversion functions
function POMDPs.convert_s(::Type{V} where V <: AbstractVector{Float64}, s::GridWorldState, mdp::LegacyGridWorld)
    v = SVector{3,Float64}(s.x, s.y, convert(Float64,s.done))
    return v
end

function POMDPs.convert_s(::Type{GridWorldState}, v::AbstractVector{Float64}, mdp::LegacyGridWorld)
    s = GridWorldState(round(Int64, v[1]), round(Int64, v[2]), convert(Bool, v[3]))
end



function test_against_full_grid()

    # Generate reward states and set to reward 10.0
    rstates = Vector{GridWorldState}(undef,0)
    rvect = Vector{Float64}(undef,0)
    for x = 40:60
        for y = 40:60
            push!(rstates,GridWorldState(x,y))
            push!(rvect,10.0)
        end
    end

    # Create full MDP - to be used by both!
    mdp = LegacyGridWorld(sx=100, sy=100, rs=rstates, rv=rvect)

    # Solve with discrete VI
    solver = ValueIterationSolver(max_iterations=1000, verbose=true)
    policy = solve(solver, mdp)

    # Setup grid with 0.1 resolution
    # As we increase VERTICES_PER_AXIS, the error should reduce
    VERTICES_PER_AXIS = 10
    grid = RectangleGrid(range(1,step=VERTICES_PER_AXIS,stop=100), range(1,step=VERTICES_PER_AXIS,stop=100), [0.0, 1.0])
    interp = LocalGIFunctionApproximator(grid)

    approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations = 1000)
    approx_policy = solve(approx_solver, mdp)


    # Randomly sample 1000 states and compute their value function match
    total_err = 0.0
    for state in states(mdp)
        full_val = value(policy,state)
        approx_val = value(approx_policy,state)
        total_err += abs(full_val-approx_val)
    end
    avg_err = total_err/10000

    println("Average difference in value function is ", avg_err)

    return true
end


@test test_against_full_grid() == true
