struct ClearDistances
    cardinal::SVector{4, Int}
    diagonal::SVector{4, Int}
end

function find_distances(f::Floor, obstacles::Set{Coord}, s::LTState)
    card = MVector{4, Int}(undef)
    diag = MVector{4, Int}(undef)
    for i in 1:4
        d = 1
        while !opaque(f, obstacles, s, s.robot+d*CARDINALS[i])
            d += 1
        end
        card[i] = d-1
    end
    for i in 1:4
        d = 1
        while !opaque(f, obstacles, s, s.robot+d*DIAGONALS[i])
            d += 1
        end
        diag[i] = d-1
    end
    return ClearDistances(card, diag)
end

mutable struct LTDistanceCache
    floor::Floor
    distances::Vector{ClearDistances}
end

function LTDistanceCache(f::Floor, obstacles::Set{Coord})
    dists = Array{ClearDistances}(undef,n_pos(f)^2)
    visited = falses(n_pos(f)^2)
    for i in 1:f.n_cols, j in 1:f.n_rows, k in 1:f.n_cols, l in 1:f.n_rows
        s = LTState(Coord(i,j), Coord(k,l), false)
        ii = stateindex(f, s)
        visited[ii] = true
        dists[ii] = find_distances(f, obstacles, s)
    end
    @assert all(visited)
    push!(dists, ClearDistances(zeros(4), zeros(4)))
    return LTDistanceCache(f, dists)
end

Base.getindex(c::LTDistanceCache, s::LTState) = c.distances[stateindex(c.floor, s)]

function n_clear_cells(d::ClearDistances, dir::Int)
    if dir <= 4
        return d.cardinal[dir]
    else
        return d.diagonal[dir-4]
    end
end

struct ReadingCDF
    cardcdf::Matrix{Float64}
    diagcdf::Matrix{Float64}
end

# reading CDF
function ReadingCDF(f::Floor,
                    std::Float64,
                    shortonly::Bool=false,
                    maxread::Int=ceil(Int, max_diag(f)+4*std))
    maxclear = max(f.n_rows, f.n_cols) - 1
    cardcdf = Array{Float64}(undef, maxclear + 1, maxread + 1)
    diagcdf = Array{Float64}(undef, maxclear + 1, maxread + 1)

    for c in 0:maxclear
        for r in 0:maxread
            cardcdf[c+1, r+1] = (1+erf((r+1.0-c)/(std*sqrt(2))))/2
        end
    end

    for c in 0:maxclear
        for r in 0:maxread
            diagcdf[c+1, r+1] = (1+erf((r+1.0-sqrt(2)*c)/(std*sqrt(2))))/2
        end
    end

    for c in 0:maxclear
        @assert abs(sum(diff(cardcdf[c, :])) - 1.0) < 1e-5
        @assert abs(sum(diff(diagcdf[c, :])) - 1.0) < 1e-5
    end

    return ReadingCDF(cardcdf, diagcdf)
end

function cdf(c::ReadingCDF, dir::Int, clear::Int, reading::Int)
    if dir <= 4 # cardinal
        if reading > size(c.cardcdf, 2)
            return 1.0
        else
            return c.cardcdf[clear + 1, reading + 1]
        end
    else # diagonal
        if reading > size(c.diagcdf, 2)
            return 1.0
        else
            return c.diagcdf[clear + 1, reading + 1]
        end
    end
end
