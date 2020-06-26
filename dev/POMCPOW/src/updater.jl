function POMDPs.updater(p::POMCPOWPlanner)
    rng = MersenneTwister(rand(p.solver.rng, UInt32))
    return SIRParticleFilter(p.problem, 10*p.solver.tree_queries, rng)
end
