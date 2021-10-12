struct NextActionSampler
    rand_act::POMCPOW.RandomActionGenerator
    b0::MEBelief
    up::MEBeliefUpdater
    function NextActionSampler(b0::MEBelief, up::MEBeliefUpdater)
        rand_act = POMCPOW.RandomActionGenerator(Random.GLOBAL_RNG)
        new(rand_act, b0, up)
    end
end

function sample_ucb_drill(mean, var)
    scores = mean[:,:,1] + sqrt.(var[:,:,1])
    weights = Float64[]
    idxs = CartesianIndex{2}[]
    m, n, _ = size(mean)
    for i =1:m
        for j = 1:n
            idx = CartesianIndex(i, j)
            push!(idxs, idx)
            push!(weights, scores[idx])
        end
    end
    coords = sample(idxs, StatsBase.Weights(weights))
    return MEAction(coords=coords)
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
            mean, var = summarize(b)
            return sample_ucb_drill(mean, var)
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
            bp = obj.b0
            s = rand(b.sr_belief.dist)[1]
            n = size(s.bore_coords)[2]
            for i = 1:n
                bore = s.bore_coords[:, i]
                coords = CartesianIndex(bore[1], bore[2])
                a = MEAction(coords=coords)
                o = MEObservation(s.ore_map[bore[1], bore[2], 1], false, false)
                bp = POMDPs.update(obj.up, bp, a, o)
            end
            mean, var = summarize(bp)
            return sample_ucb_drill(mean, var)
        end
    end
end

struct ExpertPolicy <: Policy
    m::MineralExplorationPOMDP
    # b0::MEBelief
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

mutable struct RandomSolver <: Solver
    rng::AbstractRNG
end
RandomSolver(;rng=Random.GLOBAL_RNG) = RandomSolver(rng)
POMDPs.solve(solver::RandomSolver, problem::Union{POMDP,MDP}) = POMCPOW.RandomPolicy(solver.rng, problem, BeliefUpdaters.PreviousObservationUpdater())


# function POMCPOW.estimate_value(obj::ExpertPolicy, m::MineralExplorationPOMDP,
#                         s::MEState, h::Any, ::Any)
#     bp = obj.b0
#     n = size(s.bore_coords)[2]
#     for i = 1:n
#         bore = s.bore_coords[:, i]
#         coords = CartesianIndex(bore[1], bore[2])
#         a = MEAction(coords=coords)
#         o = MEObservation(s.ore_map[bore[1], bore[2], 1], false, false)
#         bp = POMDPs.update(obj.up, bp, a, o)
#     end
#
# end
# belief_type(::MEBelief, ::MineralExplorationPOMPDP) = POWNodeBelief{statetype(P), MEBelief, obstype(P), P}
#
# init_node_sr_belief(::POWNodeFilter, p::POMDP, s, a, sp, o, r) = POWNodeBelief(p, s, a, sp, o, r)
#
# function push_weighted!(b::POWNodeBelief, ::POWNodeFilter, s, sp, r)
#     w = obs_weight(b.model, s, b.a, sp, b.o)
#     insert!(b.dist, (sp, convert(Float64, r)), w)
# end