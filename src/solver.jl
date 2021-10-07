@with_kw struct NextActionSampler
    rand_act::POMCPOW.RandomActionGenerator = POMCPOW.RandomActionGenerator(Random.GLOBAL_RNG)
end

function POMCPOW.next_action(o::NextActionSampler, pomdp::MineralExplorationPOMDP,
                            b::MEBelief, h)
    tried_idxs = h.tree.tried[h.node]
    action_set = POMDPs.actions(pomdp, b)
    if b.stopped
        if length(tried_idxs) == 0
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    else
        if MEAction(type=:stop) ∈ action_set && length(tried_idxs) <= 0
            return MEAction(type=:stop)
        else
            POMCPOW.next_action(o.rand_act, pomdp, b, h)
        end
    end
end

function POMCPOW.next_action(obj::NextActionSampler, pomdp::MineralExplorationPOMDP,
                            b::POMCPOW.StateBelief, h)
    o = b.sr_belief.o
    # s = rand(b.sr_belief.dist)[1]
    tried_idxs = h.tree.tried[h.node]
    action_set = POMDPs.actions(pomdp, b)
    if o.stopped
        if length(tried_idxs) == 0
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    else
        if MEAction(type=:stop) ∈ action_set && length(tried_idxs) <= 0
            return MEAction(type=:stop)
        else
            POMCPOW.next_action(obj.rand_act, pomdp, b, h)
        end
    end
end

struct ExpertPolicy <: Policy
    m::MineralExplorationPOMDP
end

function POMDPs.action(p::ExpertPolicy, b::MEBelief)
    if b.stopped
        volumes = Float64[]
        weights = b.particles.weights
        for s in b.particles.particles
            v = sum(s.ore_map[:, :, 1] .>= p.m.massive_threshold)
            push!(volumes, v)
        end
        mean_volume = sum(weights.*volumes)
        volume_var = sum(weights.*(volumes .- mean_volume).^2)
        volume_std = sqrt(volume_var)
        lcb = mean_volume - volume_std
        if lcb >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    else
        actions = POMDPs.actions(p.m, b)
        mean, var = summarize(b)
        mean = mean[:, : , 1]
        max_val = -Inf
        max_act = nothing
        for action in actions
            if action.type == :drill
                act_val = mean[action.coords]
                max_act = act_val > max_val ? action : max_act
                max_val = act_val > max_val ? act_val : max_val
            end
        end
        max_act = max_val > p.m.massive_threshold ? max_act : MEAction(type=:stop)
        return max_act
    end
end

function POMDPs.action(p::ExpertPolicy, b::POMCPOW.StateBelief)
    if b.stopped
        volumes = Float64[]
        weights = b.particles.weights
        for s in b.particles.particles
            v = sum(s.ore_map[:, :, 1] .>= p.m.massive_threshold)
            push!(volumes, v)
        end
        mean_volume = sum(weights.*volumes)
        volume_var = sum(weights.*(volumes .- mean_volume).^2)
        volume_std = sqrt(volume_var)
        lcb = mean_volume - volume_std
        if lcb >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    else
        actions = POMDPs.actions(p.m, b)
        mean, var = summarize(b)
        mean = mean[:, : , 1]
        max_val = -Inf
        max_act = nothing
        for action in actions
            if action.type == :drill
                act_val = mean[action.coords]
                max_act = act_val > max_val ? action : max_act
                max_val = act_val > max_val ? act_val : max_val
            end
        end
        max_act = max_val > p.m.massive_threshold ? max_act : MEAction(type=:stop)
        return max_act
    end
end


"""
solver that produces a random policy
"""
mutable struct RandomSolver <: Solver
    rng::AbstractRNG
end
RandomSolver(;rng=Random.GLOBAL_RNG) = RandomSolver(rng)
POMDPs.solve(solver::RandomSolver, problem::Union{POMDP,MDP}) = POMCPOW.RandomPolicy(solver.rng, problem, BeliefUpdaters.PreviousObservationUpdater())
