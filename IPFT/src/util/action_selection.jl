abstract type AbstractMultiObjectiveCriterion end

struct MultiObjectiveUCB <: AbstractMultiObjectiveCriterion
    weights::Vector{Float64}
    c::Float64
end

q_value(q_vec::Array{Float64}, crit::AbstractMultiObjectiveCriterion) = dot(crit.weights, q_vec)

function select_best(crit::MultiObjectiveUCB, node::AbstractStateNode, rng::AbstractRNG)
    best_UCB = -Inf
    sanode = 0
    ltn = log(total_n(node))
    for child in children(node)
        n = node.tree.n[child]
        wq = q_value(node.tree.q[child], crit)
        if (ltn <= 0 && n == 0) || crit.c == 0.0
            UCB = wq
        else
            UCB = wq + crit.c*sqrt(ltn/n)
        end
        @assert !isnan(UCB) "UCB was NaN (wq=$wq, q_vec=$(node.tree.q[child]), c=$(crit.c), ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end
    sanode
end

struct MultiObjective <: AbstractMultiObjectiveCriterion
    weights::Vector{Float64}
end
MultiObjective(crit::MultiObjectiveUCB) = MultiObjective(crit.weights)

function select_best(crit::MultiObjective, node::AbstractStateNode, rng::AbstractRNG)
    best_Q = -Inf
    sanode = 0
    for child in children(node)
        wq = q_value(node.tree.q[child], crit)
        if wq > best_Q
            best_Q = wq
            sanode = child
        end
    end
    sanode
end

struct MaxTries <: AbstractMultiObjectiveCriterion end

function select_best(crit::MaxTries, node::AbstractStateNode, rng::AbstractRNG)
    best_child = first(children(node))
    best_n = total_n(best_child)
    @assert !isnan(best_n)
    for child in children(node)[2:end]
        if total_n(child) >= best_n
            best_n = total_n(child)
            best_child = child
        end
    end
    return best_child
end
