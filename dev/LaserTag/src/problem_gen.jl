function cpp_emu_lasertag(r::Int; kwargs...)
    if r == 4
        obstacles = Set{Coord}(Coord(c) for c in ([5,7], [2,6], [4,3], [3,2], [4,2], [8,2], [10,2], [4,1]))
    else
        warn("r = $r not recognized.")
    end
    return gen_lasertag(7, 11; obstacles=obstacles, kwargs...)
end

#=
function gen_lasertag(n_rows::Int,
                      n_cols::Int,
                      n_obstacles::Int,
                      reading_std::Float64=2.5;
                      discrete=false,
                      rng=Base.GLOBAL_RNG,
                      kwargs...)

    return gen_lasertag(n_rows, n_cols;
                        obstacles=gen_obstacles(n_rows, n_cols, n_obstacles, rng),
                        discrete=discrete,
                        reading_std=reading_std,
                        rng=rng,
                        kwargs...
                       )
end
=#

function gen_lasertag(n_rows::Int,
                      n_cols::Int,
                      n_obstacles::Int,
                      obs_model::ObsModel=DESPOTEmu(Floor(n_rows, n_cols), 2.5);
                      rng=Random.GLOBAL_RNG,
                      kwargs...
                     )
    
    return gen_lasertag(n_rows,
                        n_cols,
                        obstacles=gen_obstacles(n_rows, n_cols, n_obstacles, rng),
                        obs_model=obs_model,
                        rng=rng;
                        kwargs...
                       )
end

function gen_obstacles(n_rows::Int, n_cols::Int, n_obstacles::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    f = Floor(n_rows, n_cols)
    obs_inds = randperm(rng, n_pos(f))[1:n_obstacles] # XXX inefficient
    obs_subs = CartesianIndices((n_cols, n_rows))[obs_inds]
    obstacles = Set{Coord}(Coord(p.I) for p in obs_subs)
    for c in obstacles
        if !inside(f, c)
            @show c
            @show obs_inds
            @show obs_subs
            error("not inside")
        end
    end
    return obstacles
end

function gen_lasertag(n_rows::Int=7,
                      n_cols::Int=11;
                      rng=Random.GLOBAL_RNG,
                      obstacles=gen_obstacles(n_rows, n_cols, 8, rng),
                      obs_model::ObsModel=DESPOTEmu(Floor(n_rows, n_cols), 2.5),
                      robot_position_known::Bool=false,
                      kwargs...)

    f = Floor(n_rows, n_cols)
    if robot_position_known
        r = Coord(rand(rng, 1:f.n_cols), rand(rng, 1:f.n_rows))
    else
        r = nothing
    end
    M = typeof(obs_model)
    O = obs_type(M)
    return LaserTagPOMDP{M, O}(;floor=f,
                                obstacles=obstacles,
                                robot_init=r, 
                                obs_model=obs_model,
                                kwargs...)
end
