@with_kw struct NextActionSampler
    ucb::Float64 = 1.0
    # rand_act::POMCPOW.RandomActionGenerator
    # b0::MEBelief
    # up::MEBeliefUpdater
    # function NextActionSampler(b0::MEBelief, up::MEBeliefUpdater)
    #     rand_act = POMCPOW.RandomActionGenerator(Random.GLOBAL_RNG)
    #     new(rand_act, b0, up)
    # end
end

function sample_ucb_drill(mean, var)
    scores = belief_scores(mean, var)
    weights = Float64[]
    idxs = CartesianIndex{2}[]
    m, n, _ = size(mean)
    for i =1:m
        for j = 1:n
            idx = CartesianIndex(i, j)
            push!(idxs, idx)
            push!(weights, scores[i, j])
        end
    end
    coords = sample(idxs, StatsBase.Weights(weights))
    return MEAction(coords=coords)
end

function belief_scores(m, v)
    norm_mean = m[:,:,1]./(maximum(m[:,:,1]) - minimum(m[:,:,1]))
    norm_mean .-= minimum(norm_mean)
    s = v[:,:,1]
    norm_std = s./(maximum(s) - minimum(s)) # actualy using variance
    norm_std .-= minimum(norm_std)
    scores = norm_mean .* norm_std
    scores ./= sum(scores)
    return scores
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
            ore_maps = Array{Float64, 3}[]
            weights = Float64[]
            for (idx, item) in enumerate(b.sr_belief.dist.items)
                weight = b.sr_belief.dist.cdf[idx]
                state = item[1]
                push!(ore_maps, state.ore_map)
                push!(weights, weight)
            end
            weights ./= sum(weights)
            mean = sum(weights.*ore_maps)
            var = sum([weights[i]*(ore_map - mean).^2 for (i, ore_map) in enumerate(ore_maps)])
            return sample_ucb_drill(mean, var)
        end
    end
end

struct ExpertPolicy <: Policy
    m::MineralExplorationPOMDP
end

POMCPOW.updater(p::ExpertPolicy) = MEBeliefUpdater(p.m, 0)

function POMCPOW.BasicPOMCP.extract_belief(p::MEBeliefUpdater, node::POMCPOW.BeliefNode)
    srb = node.tree.sr_beliefs[node.node]
    cv = srb.dist
    particles = MEState[]
    weights = Float64[]
    coords = nothing
    stopped = false
    for (idx, item) in enumerate(cv.items)
        weight = cv.cdf[idx]
        state = item[1]
        coords = state.bore_coords
        stopped = state.stopped
        push!(particles, state)
        push!(weights, weight)
    end
    return MEBelief(coords, stopped, particles, MEAction[], MEObservation[])
end

function POMDPs.action(p::ExpertPolicy, b::MEBelief)
    if b.stopped
        volumes = Float64[]
        # weights = b.weights
        for s in b.particles
            v = sum(s.ore_map[:, :, 1] .>= p.m.massive_threshold)
            push!(volumes, v)
        end
        # mean_volume = sum(weights.*volumes)
        mean_volume = Statistics.mean(volumes)
        # volume_var = sum(weights.*(volumes .- mean_volume).^2)
        volume_var = Statistics.var(volumes)
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
        weights = b.weights
        for s in b.particles
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
