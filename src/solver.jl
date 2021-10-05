@with_kw struct NextActionSampler
    rand_act::POMCPOW.RandomActionGenerator = POMCPOW.RandomActionGenerator(Random.GLOBAL_RNG)
end

function POMCPOW.next_action(o::NextActionSampler, pomdp, b, h)
    tried_idxs = h.tree.tried[h.node]
    if length(tried_idxs) <= 0
        return :stop
    else
        POMCPOW.next_action(o.rand_act, pomdp, b, h)
    end
end

struct ExpertPolicy <: Policy
    m::MineralExplorationPOMDP
end

function POMDPs.action(p::ExpertPolicy, b::MEBelief)
    actions = POMDPs.actions(p.m, b)
    mean, var = summarize(b)
    mean = mean[:, : , 1]
    max_val = -Inf
    max_act = nothing
    for action in actions
        if action != :stop
            act_val = mean[action]
            max_act = act_val > max_val ? action : max_act
            max_val = act_val > max_val ? act_val : max_val
        end
    end
    max_act = max_val > p.m.massive_threshold ? max_act : :stop
    return max_act
end
