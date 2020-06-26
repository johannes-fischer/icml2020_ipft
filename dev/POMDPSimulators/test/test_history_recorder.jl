let
problem = BabyPOMDP()
policy = RandomPolicy(problem, rng=MersenneTwister(2))
steps=10
sim = HistoryRecorder(max_steps=steps, rng=MersenneTwister(3))
@show_requirements simulate(sim, problem, policy, updater(policy), initialstate_distribution(problem))
r1 = simulate(sim, problem, policy, updater(policy), initialstate_distribution(problem))
policy.rng = MersenneTwister(2)
sim.rng = MersenneTwister(3)
r2 = simulate(sim, problem, policy)

@test length(state_hist(r1)) == steps+1
@test length(action_hist(r1)) == steps
@test length(observation_hist(r1)) == steps
@test length(belief_hist(r1)) == steps+1
@test length(state_hist(r2)) == steps+1
@test length(action_hist(r2)) == steps
@test length(observation_hist(r2)) == steps
@test length(belief_hist(r2)) == steps+1
@test length(info_hist(r2)) == steps
@test length(ainfo_hist(r2)) == steps
@test length(uinfo_hist(r2)) == steps

@test exception(r1) == nothing
@test exception(r2) == nothing
@test backtrace(r1) == nothing
@test backtrace(r2) == nothing

@test n_steps(r1) == n_steps(r2)
@test undiscounted_reward(r1) == undiscounted_reward(r2)
@test discounted_reward(r1) == discounted_reward(r2)

@test length(collect(r1)) == n_steps(r1)
@test length(collect(r2)) == n_steps(r2)

for tuple in r1
    length(tuple) == 6
end

for ui in eachstep(r2, "ui")
    @test ui == nothing
end

# test that complete step is returned
step = first(eachstep(r2))
for key in POMDPSimulators.COMPLETE_POMDP_STEP
    @test haskey(step, key)
end

problem = LegacyGridWorld()
policy = RandomPolicy(problem, rng=MersenneTwister(2))
steps=10
sim = HistoryRecorder(max_steps=steps, rng=MersenneTwister(3))
@show_requirements simulate(sim, problem, policy, initialstate(problem, sim.rng))
r1 = simulate(sim, problem, policy, initialstate(problem, sim.rng))

@test length(state_hist(r1)) <= steps + 1 # less than or equal because it may reach the goal too fast
@test length(action_hist(r1)) <= steps
@test length(reward_hist(r1)) <= steps

for tuple in r1
    @test length(tuple) == 4
    @test isa(tuple[1], statetype(problem))
    @test isa(tuple[2], actiontype(problem))
    @test isa(tuple[3], Float64)
    @test isa(tuple[4], statetype(problem))
    @test isa(tuple.s, statetype(problem))
    @test isa(tuple.a, actiontype(problem))
    @test isa(tuple.r, Float64)
    @test isa(tuple.sp, statetype(problem))
end

@test length(collect(r1)) == n_steps(r1)

hv = view(r1, 2:length(r1))
@test n_steps(hv) == n_steps(r1)-1
@test undiscounted_reward(r1) == undiscounted_reward(hv) + reward_hist(r1)[1]

# iterators
rsum = 0.0
len = 0
for (s, a, r, sp, i, ai, t) in eachstep(hv, (:s,:a,:r,:sp,:i,:ai,:t))
    @test isa(s, statetype(problem))
    @test isa(a, actiontype(problem))
    @test isa(r, Float64)
    @test isa(sp, statetype(problem))
    @test isa(i, Nothing)
    @test isa(ai, Nothing)
    @test isa(t, Int)
    rsum += r
    len += 1
end
@test len == length(hv)
@test rsum == undiscounted_reward(hv)

# it = eachstep(hv, "(r,sp,s,a)")
# @test eltype(collect(it)) == Tuple{Float64, statetype(problem), statetype(problem), actiontype(problem)}
tuples = collect(eachstep(hv, "(r, sp, s, a)"))
@test sum(first(t) for t in tuples) == undiscounted_reward(hv)
@test sum(t.r for t in tuples) == undiscounted_reward(hv)
tuples = collect(eachstep(hv, "r,sp,s,a,t"))
@test sum(first(t) for t in tuples) == undiscounted_reward(hv)
@test sum(t.r for t in tuples) == undiscounted_reward(hv)

@test collect(eachstep(hv, "r")) == reward_hist(hv)

#=
function f(hv)
    rs = 0.0
    for (r,a) in HistoryIterator{typeof(hv), (:r,:a)}(hv)
        rs += r
    end
    return rs
end
@code_warntype f(hv)
hi = HistoryIterator{typeof(r1), (:r,)}(r1)
t = step_tuple(hi, 1)
@code_warntype step_tuple(hi, 1)
=#
end
