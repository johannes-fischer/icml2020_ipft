struct IBMDPSolver <: Solver
    ipft::IPFTSolver
    updater::Union{Function, Updater}
    ifm::InformationMeasure
    discount_information_gain::Bool
end

function POMDPs.solve(solver::IBMDPSolver, pomdp::POMDP)
    if isa(solver.updater, Function)
        updater = solver.updater(pomdp)
    else
        updater = solver.updater
    end
    belief_mdp = InformationRewardBeliefMDP(deepcopy(pomdp),
                                     updater.resample,
                                     solver.ifm,
                                     solver.discount_information_gain,
                                     updater.max_frac_replaced
                                    )
    return solve(solver.ipft, belief_mdp)
end

function POMDPs.solve(qs::QMDPSolver, bmdp::InformationRewardBeliefMDP)
    return solve(qs, bmdp.pomdp)
end
