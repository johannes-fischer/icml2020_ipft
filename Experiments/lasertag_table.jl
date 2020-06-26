filename = @__FILE__()
file_dir = @__DIR__()
file_contents = read(filename, String)

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--pknown"
        arg_type = Bool
        help = "robot position known or unknown"
        default = true
    "-N"
        help = "number of simulations per solver"
        arg_type = Int
        default = 1000
    "--test"
        help = "flag for using only one heuristic solver and N=1"
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
using LaserTag


seed = Random.GLOBAL_RNG.seed[1]
file_contents = "rng seed $seed \n" * repr("text/plain", parsed_args)* "\n" * file_contents

@show max_time = 1.0
@show max_depth = 90
@show exploration = 26.0
N = parsed_args["test"] ? 1 : parsed_args["N"]
@show N
# ro = 1
pos_known = parsed_args["pknown"]

template_problem = gen_lasertag()


solvers = Dict{String, Union{Solver,Policy}}(

    # "qmdp" => QMDPSolver(),

    "pomcpow" => begin
        rng = MersenneTwister(seed)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(exploration),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=4.0,
                               alpha_observation=1/35,
                               estimate_value=FOValue(ValueIterationSolver()),
                               check_repeat_obs=false,
                               tree_in_info=false,
                               default_action=LaserTag.TAG_ACTION,
                               rng=rng
                              )
    end,

    "pft" => begin
        rng = MersenneTwister(seed)
        m = 20
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=exploration,
                           depth=max_depth,
                           max_time=max_time,
                           k_state=4.0,
                           alpha_state=1/35,
                           check_repeat_state=false,
                           check_repeat_action=true,
                           tree_in_info=false,
                           estimate_value=RolloutEstimator(QMDPSolver()),
                           enable_action_pw=false,
                           # default_action=ReportWhenUsed(NoGapTag()),
                           rng=rng
                          )
        GBMDPSolver(solver, pomdp->ObsAdaptiveParticleFilter(pomdp,
                                                             LowVarianceResampler(m),
                                                             0.1, rng))
    end,

    "ipft-1" => begin
        rng = MersenneTwister(seed)
        m = 20
        ifm = DiscreteEntropy()
        discount_information_gain = false
        solver = IPFTSolver(n_iterations=typemax(Int),
                        criterion=MultiObjectiveUCB([1.0, 4.0], exploration),
                        depth=max_depth,
                        max_time=max_time,
                        check_repeat_action=true,
                        reuse_states=false,
                        k_observation = 4.0,
                        alpha_observation = 1/35,
                        estimate_value=FOValue(QMDPSolver()),
                        enable_action_pw=false,
                        tree_in_info=true,
                        rng=rng
                        )
        IBMDPSolver(solver, pomdp->ObsAdaptiveParticleFilter(pomdp,
                                                             LowVarianceResampler(m),
                                                             0.1, rng),
                                                             ifm, discount_information_gain)

    end,

    "ipft-gamma" => begin
        rng = MersenneTwister(seed)
        m = 20
        ifm = DiscreteEntropy()
        discount_information_gain = true
        solver = IPFTSolver(n_iterations=typemax(Int),
                        criterion=MultiObjectiveUCB([1.0, 4.0], exploration),
                        depth=max_depth,
                        max_time=max_time,
                        check_repeat_action=true,
                        reuse_states=false,
                        k_observation = 4.0,
                        alpha_observation = 1/35,
                        estimate_value=FOValue(QMDPSolver()),
                        enable_action_pw=false,
                        tree_in_info=true,
                        rng=rng
                        )
        IBMDPSolver(solver, pomdp->ObsAdaptiveParticleFilter(pomdp,
                                                             LowVarianceResampler(m),
                                                             0.1, rng),
                                                             ifm, discount_information_gain)

    end,

)

# Consider not all solvers, depending on parameters
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
    # prog = Progress(N, desc="Creating Simulations...")
    # sims = pmap(prog, 1:N) do i
    sims = []

    for i=1:N
        # sample a different problem in every run (randomly initialized)
        pomdp = gen_lasertag(rng=MersenneTwister(seed+i+30_000), robot_position_known=pos_known)
        if isa(solver, Solver)
            planner = solve(solver, pomdp)
        else
            planner = solver
        end

        be_updater = planner isa InformationMCTSPlanner

        Random.seed!(planner, seed+i+500_000)
        up_rng = MersenneTwister(seed+i+240_000)
        up = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(10_000), 0.05, up_rng)

        if be_updater
            up = InformationFilter(up, planner.mdp.im)
        end

        md = Dict(:solver=>k, :i=>i)
        sim = Sim(pomdp,
            planner,
            up,
            rng=MersenneTwister(seed+i+50_000),
            max_steps=100,
            metadata=md
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
