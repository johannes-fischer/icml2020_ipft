function reset_distribution(p::LaserTagPOMDP, b::ParticleCollection, a, o)
    # warn("Resetting Particle Filter Distribution")
    rob = first(particles(b)).robot
    nextrob = LaserTag.add_if_inside(p.floor, rob, LaserTag.ACTION_DIRS[a])
    if o == LaserTag.C_SAME_LOC
        return ParticleCollection{LaserTag.LTState}([LaserTag.LTState(nextrob, nextrob, false)])
    else
        return LaserTag.LTInitialBelief(nextrob, p.floor)
    end
end

max_possible_weight(pomdp::LaserTagPOMDP, a::Int, o) = 0.0

new_particle(pomdp::LaserTagPOMDP, a::Int, o) = error("tried to generate a new particle (shouldn't get here)")


# max_possible_weight(pomdp::LaserTagPOMDP, a::Int, o::Float64) = max(1.0, pdf(Normal(0.0, pomdp.return_std), 0.0))
