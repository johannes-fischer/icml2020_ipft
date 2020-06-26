filename = @__FILE__()
file_dir = @__DIR__()
file_contents = read(filename, String)

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--step", "-s"
        default = 10
        arg_type = Int
        help = "big step for problem definition"
    "--obs", "-o"
        help = "observation noise added to distance dependent noise"
        arg_type = Float64
        default = 0.0001
    "-N"
        help = "number of simulations per solver"
        arg_type = Int
        default = 1000
    "--test"
        help = "flag for using only one solver and N=1"
        action = :store_true
    "--solver"
        help = "select one solver to run by name"
        arg_type = String
end
parsed_args = parse_args(ARGS, s)

using DataFrames
using CSV
using Printf
using Random
using Dates

using POMDPs
using ParticleFilters
using MCTS
using POMCPOW
using QMDP
using DiscreteValueIteration
using POMDPSimulators
using POMDPPolicies

using SunbergTypes
using IPFT
using Scenarios



seed = Random.GLOBAL_RNG.seed[1]
file_contents = "rng seed $seed \n" * repr("text/plain", parsed_args) * "\n" * file_contents

@show max_time = 1.0
@show max_depth = 20
N = parsed_args["test"] ? 1 : parsed_args["N"]
@show N

pomdp = SimpleLightDark(big_step=parsed_args["step"], min_obs_noise=parsed_args["obs"])
@show pomdp

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rng = MersenneTwister(seed)
        ro = ValueIterationSolver()
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(90.0),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=5.,
                               alpha_observation=1/15.0,
                               estimate_value=FOValue(ro),
                               check_repeat_obs=false,
                               tree_in_info=false,
                               rng=rng
                              )
        solver
    end,

    "pft" => begin
        rng = MersenneTwister(seed)
        m = 20
        ro = solve(QMDPSolver(), pomdp)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=100.0,
                           depth=max_depth,
                           max_time=max_time,
                           k_state=4.0,
                           alpha_state=1/10,
                           check_repeat_state=false,
                           estimate_value=RolloutEstimator(ro),
                           enable_action_pw=false,
                           tree_in_info=false,
                           rng=rng
                          )
        belief_mdp = MeanRewardBeliefMDP(pomdp,
                                         LowVarianceResampler(m),
                                         0.05
                                        )

        solve(solver, belief_mdp)
    end,

    "ipft-1" => begin
        rng = MersenneTwister(seed)
        m = 20
        ro = solve(QMDPSolver(), pomdp)
        ifm = DiscreteEntropy()
        discount_information_gain = false
        solver = IPFTSolver(n_iterations=typemax(Int),
                        criterion=MultiObjectiveUCB([1.0,50.0],100.0),
                        depth=max_depth,
                        max_time=max_time,
                        check_repeat_action=true,
                        reuse_states=true,
                        k_observation = 5.0,
                        alpha_observation = 1/20,
                        estimate_value=RolloutEstimator(ro),
                        enable_action_pw=false,
                        tree_in_info=true,
                        rng=rng
                        )
        belief_mdp = InformationRewardBeliefMDP(pomdp,
                                         LowVarianceResampler(m),
                                         ifm,
                                         discount_information_gain,
                                         0.05
                                        )
        solve(solver, belief_mdp)
    end,

    "ipft-gamma" => begin
        rng = MersenneTwister(seed)
        m = 20
        ro = solve(QMDPSolver(), pomdp)
        ifm = DiscreteEntropy()
        discount_information_gain = true
        solver = IPFTSolver(n_iterations=typemax(Int),
                        criterion=MultiObjectiveUCB([1.0,50.0],100.0),
                        depth=max_depth,
                        max_time=max_time,
                        check_repeat_action=true,
                        reuse_states=true,
                        k_observation = 5.0,
                        alpha_observation = 1/20,
                        estimate_value=RolloutEstimator(ro),
                        enable_action_pw=false,
                        tree_in_info=true,
                        rng=rng
                        )
        belief_mdp = InformationRewardBeliefMDP(pomdp,
                                         LowVarianceResampler(m),
                                         ifm,
                                         discount_information_gain,
                                         0.05
                                        )
        solve(solver, belief_mdp)
    end,

)

# Consider not all solvers, depending on arguments
if parsed_args["test"]
    solvers = Dict(rand(solvers))
elseif !(parsed_args["solver"] isa Nothing)
    name = parsed_args["solver"]
    solvers = Dict(name=>solvers[name])
end

alldata = DataFrame()
for (k, solver) in solvers
    global alldata
    @show k
    if isa(solver, Solver)
        planner = solve(solver, pomdp)
    else
        planner = solver
    end
    sims = []

    be_updater = planner isa InformationMCTSPlanner

    for i in 1:N
        Random.seed!(planner, seed+i+50_000)

        # up = SIRParticleFilter(pomdp, 10_000, MersenneTwister(seed+i+90_000))
        up = ObsAdaptiveParticleFilter(pomdp,
                                           LowVarianceResampler(10_000),
                                           0.05, MersenneTwister(seed+i+90_000))
        if be_updater
            up = InformationFilter(up, planner.mdp.im)
        end

        sim = Sim(pomdp,
                  planner,
                  up,
                  rng=MersenneTwister(seed+i+70_000),
                  max_steps=100,
                  metadata=Dict(:solver=>k, :i=>i)
                 )

        push!(sims, sim)
    end

    data = run_parallel(sims)
    # data = run(sims)

    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
    if isempty(alldata)
        alldata = data
    else
        alldata = vcat(alldata, data)
    end
end

# Filename without path
filename = split(filename,'/')[end]
# Filename without extension
filename = split(filename,'.')[1]
# Get rid of the _table suffix
filename = join(split(filename,'_')[1:max(1,end-1)], '_')
# Append problem parameters to filename
filename = filename * repr(parsed_args["step"])

result_dir = joinpath(file_dir, "results")
if !isdir(result_dir)
    mkdir(result_dir)
end

datestring = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
copyname = joinpath(result_dir, filename * "_table_$(datestring).jl")
write(copyname, file_contents)
csvname = joinpath(result_dir, filename * "_$(datestring).csv")
println("saving to $csvname...")
CSV.write(csvname, alldata)
println("done.")
