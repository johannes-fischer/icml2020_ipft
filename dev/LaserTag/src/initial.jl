struct LTInitialBelief
    robot_init::Union{Coord, Nothing}
    floor::Floor
end

sampletype(::Type{LTInitialBelief}) = LTState
function iterator(b::LTInitialBelief)
    states = LTState[]
    for (x,y) in product(1:b.floor.n_cols, 1:b.floor.n_rows)
        if isnull(b.robot_init)
            for (rx, ry) in product(1:b.floor.n_cols, 1:b.floor.n_rows)
                push!(states, LTState(Coord(rx,ry), Coord(x,y), false))
            end
        else
            push!(states, LTState(get(b.robot_init), Coord(x,y), false))
        end
    end
    return states
end

function Random.rand(rng::AbstractRNG, b::LTInitialBelief)
    opp = Coord(rand(rng, 1:b.floor.n_cols), rand(rng, 1:b.floor.n_rows))
    if b.robot_init === nothing
        rob = Coord(rand(rng, 1:b.floor.n_cols), rand(rng, 1:b.floor.n_rows))
    else
        rob = b.robot_init
    end
    return LTState(rob, opp, false)
end

function Distributions.pdf(b::LTInitialBelief, s::LTState)
    if s.terminal
        return 0.0
    else
        if b.robot_init === nothing
            return (1/n_pos(b.floor))^2
        else
            if s.robot == b.robot_init
                return 1/n_pos(b.floor)
            else
                return 0.0
            end
        end
    end
end

POMDPSimulators.initialstate_distribution(p::LaserTagPOMDP) = LTInitialBelief(p.robot_init, p.floor)
