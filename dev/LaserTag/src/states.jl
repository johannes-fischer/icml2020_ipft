POMDPs.states(p::LaserTagPOMDP) = states(p.floor)

function states(f::Floor) # inefficient?
    vec = Array{LTState}(undef, n_pos(f)^2+1)
    for i in 1:f.n_cols, j in 1:f.n_rows, k in 1:f.n_cols, l in 1:f.n_rows
        s = LTState(Coord(i,j), Coord(k,l), false)
        vec[stateindex(f, s)] = s
    end
    vec[end] = LTState(Coord(1,1), Coord(1,1), true)    
    return vec
end

n_pos(f::Floor) = f.n_rows*f.n_cols

POMDPs.n_states(p::LaserTagPOMDP) = n_pos(p.floor)^2+1

POMDPs.stateindex(p::LaserTagPOMDP, s::LTState) = stateindex(p.floor, s)

function stateindex(f::Floor, s::LTState)
    nr = f.n_rows
    nc = f.n_cols
    n_pos = nr^2*nc^2
    if s.terminal
        return n_pos+1
    else
        rob = s.robot
        opp = s.opponent
        return LinearIndices((nc,nr,nc,nr))[rob[1], rob[2], opp[1], opp[2]]
    end
end
