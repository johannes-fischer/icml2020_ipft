using POMDPModels
using Test

let
    problem = BabyPOMDP()
    solver = RandomSolver(rng=Random.MersenneTwister(1))
    policy = solve(solver, problem)
    sim = RolloutSimulator(max_steps=10, rng=Random.MersenneTwister(1))
    r1 = @inferred simulate(sim, problem, policy, updater(policy), initialstate_distribution(problem))

    sim = RolloutSimulator(max_steps=10, rng=Random.MersenneTwister(1))
    dummy = @inferred simulate(sim, problem, policy, updater(policy), nothing, true)

    problem = LegacyGridWorld()
    solver = RandomSolver(rng=Random.MersenneTwister(1))
    policy = solve(solver, problem)
    sim = RolloutSimulator(max_steps=10, rng=Random.MersenneTwister(1))
    r2 = @inferred simulate(sim, problem, policy, initialstate(problem, sim.rng))

    problem = LegacyGridWorld()
    solver = RandomSolver(rng=Random.MersenneTwister(1))
    policy = solve(solver, problem)
    sim = RolloutSimulator(Random.MersenneTwister(1), 10) # new constructor
    r2 = @inferred simulate(sim, problem, policy, initialstate(problem, sim.rng))

    @test isapprox(r1, -27.27829, atol=1e-3)
    @test isapprox(r2, 0.0, atol=1e-3)
end
