POMDPs.n_actions(p::LaserTagPOMDP) = p.diag_actions ? 9 : 5
POMDPs.actions(p::LaserTagPOMDP) = 1:n_actions(p)
POMDPs.actionindex(p::LaserTagPOMDP, a::Int) = findfirst(actions(p) .== a)

const ACTION_NAMES = SVector("north",
                             "east",
                             "south",
                             "west",
                             "tag",
                             "northeast",
                             "southeast",
                             "southwest",
                             "northwest")

const TAG_ACTION = 5

const ACTION_DIRS = SVector(Coord(0,1),
                            Coord(1,0),
                            Coord(0,-1),
                            Coord(-1,0),
                            Coord(0,0),
                            Coord(1,1),
                            Coord(1,-1),
                            Coord(-1,-1),
                            Coord(-1,1))

const CARDINALS = SVector(Coord(0,1),
                            Coord(1,0),
                            Coord(0,-1),
                            Coord(-1,0))

const DIAGONALS = SVector(Coord(1,1),
                         Coord(1,-1),
                         Coord(-1,-1),
                         Coord(-1,1))

const DIR_TO_ACTION = Dict{Coord, Int}(c=>i for (i,c) in enumerate(ACTION_DIRS))
