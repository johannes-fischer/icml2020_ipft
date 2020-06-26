using POMDPs
using LaserTag
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
using ProfileView
using Random
using Profile

p = gen_lasertag(3,4,2)

pol = RandomPolicy(p, rng=MersenneTwister(1))

sim = RolloutSimulator(max_steps=100, rng=MersenneTwister(2))

fltr = SIRParticleFilter(p, 10000)

@time simulate(sim, p, pol, fltr)
@time simulate(sim, p, pol, fltr)
simulate(sim, p, pol, fltr)

Profile.clear()
@profile simulate(sim, p, pol, fltr)

ProfileView.view()
