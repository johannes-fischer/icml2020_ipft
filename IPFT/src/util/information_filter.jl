struct InformationFilter <: Updater
    up::Updater
    im::InformationMeasure
    initial_im::InformationMeasure
end
InformationFilter(up::Updater, im::InformationMeasure) = InformationFilter(up, im, im)

function POMDPs.update(up::InformationFilter, ib::InformationBelief{B}, a, o) where {B <: ParticleCollection}
    @debug("Start entropy filter update")
    b = ib.b
    bp = update(up.up, b, a, o)
    wbp = weighted_posterior_belief(up.up)
    bpred = predicted_belief(up.up)
    i = information(up.im, bp=bp, wbp=wbp, b=b, bpred=bpred, a=a, o=o)
    InformationBelief(bp, i)
end

function POMDPs.initialize_belief(up::InformationFilter, d::Any)
    b = initialize_belief(up.up, d)
    i = information(up.initial_im, bp=b)
    InformationBelief(b, i)
end
