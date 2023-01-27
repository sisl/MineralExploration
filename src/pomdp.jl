function GeoStatsDistribution(p::MineralExplorationPOMDP; truth=false)
    grid_dims = truth ? p.high_fidelity_dim : p.grid_dim
    variogram = SphericalVariogram(sill=p.variogram[1], range=p.variogram[2],
                                    nugget=p.variogram[3])
    domain = CartesianGrid{Int64}(grid_dims[1], grid_dims[2])
    #return GeoStatsDistribution(rng=p.rng,
    return GeoStatsDistribution(grid_dims=grid_dims,
                                data=deepcopy(p.initial_data),
                                domain=domain,
                                mean=p.gp_mean,
                                variogram=variogram)
end

function GSLIBDistribution(p::MineralExplorationPOMDP)
    variogram = (1, 1, 0.0, 0.0, 0.0, p.variogram[2], p.variogram[2], 1.0)
    # variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    return GSLIBDistribution(grid_dims=p.grid_dim, n=p.grid_dim,
                            data=deepcopy(p.initial_data), mean=p.gp_mean,
                            sill=p.variogram[1], variogram=variogram,
                            nugget=p.variogram[3])
end

"""
    sample_coords(dims::Tuple{Int, Int}, n::Int)
Sample coordinates from a Cartesian grid of dimensions given by dims and return
them in an array
"""
function sample_coords(dims::Tuple{Int, Int, Int}, n::Int)
    idxs = CartesianIndices(dims)
    samples = sample(idxs, n)
    sample_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(samples)
        sample_array[1, i] = sample[1]
        sample_array[2, i] = sample[2]
    end
    return (samples, sample_array)
end

function sample_initial(p::MineralExplorationPOMDP, n::Integer)
    coords, coords_array = sample_coords(p.grid_dim, n)
    dist = GeoStatsDistribution(p)
    state = rand(p.rng, dist)
    ore_quality = state[coords]
    return RockObservations(ore_quality, coords_array)
end

function sample_initial(p::MineralExplorationPOMDP, coords::Array)
    n = length(coords)
    coords_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(coords)
        coords_array[1, i] = sample[1]
        coords_array[2, i] = sample[2]
    end
    dist = GeoStatsDistribution(p)
    state = rand(p.rng, dist)
    ore_quality = state[coords]
    return RockObservations(ore_quality, coords_array)
end

function initialize_data!(p::MineralExplorationPOMDP, n::Integer)
    new_rock_obs = sample_initial(p, n)
    append!(p.initial_data.ore_quals, new_rock_obs.ore_quals)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

function initialize_data!(p::MineralExplorationPOMDP, coords::Array)
    new_rock_obs = sample_initial(p, coords)
    append!(p.initial_data.ore_quals, new_rock_obs.ore_quals)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

POMDPs.discount(::MineralExplorationPOMDP) = 0.99
POMDPs.isterminal(m::MineralExplorationPOMDP, s::MEState) = s.decided

function POMDPs.initialstate_distribution(m::MineralExplorationPOMDP)
    true_gp_dist = m.geodist_type(m; truth=true)
    gp_dist = m.geodist_type(m)
    MEInitStateDist(true_gp_dist, gp_dist, m.mainbody_weight,
                    m.true_mainbody_gen, m.mainbody_gen,
                    m.massive_threshold, m.dim_scale, m.target_dim_scale,
                    m.target_mass_params[1], m.target_mass_params[2], m.rng) #m.rng passes global
end

function Base.rand(rng::Random.AbstractRNG, d::MEInitStateDist, n::Int=1; truth::Bool=false, apply_scale::Bool=false)
    gp_dist = truth ? d.true_gp_distribution : d.gp_distribution
    gp_ore_maps = Base.rand(rng, gp_dist, n)
    if n == 1
        gp_ore_maps = Array{Float64, 3}[gp_ore_maps]
    end

    states = MEState[]
    x_dim = gp_dist.grid_dims[1]
    y_dim = gp_dist.grid_dims[2]
    mainbody_gen = truth ? d.true_mainbody_gen : d.mainbody_gen
    for i = 1:n
        lode_map, lode_params = rand(rng, mainbody_gen)
        lode_map = normalize_and_weight(lode_map, d.mainbody_weight)

        gp_ore_map = gp_ore_maps[i]
        ore_map = lode_map + gp_ore_map
        if apply_scale
            ore_map, lode_params = scale_sample(d, mainbody_gen, lode_map, gp_ore_map, lode_params; target_μ=d.target_μ, target_σ=d.target_σ)
        end
        state = MEState(ore_map, lode_params, lode_map,
                RockObservations(), false, false)
        push!(states, state)
    end
    if n == 1
        return states[1]
    else
        return states
    end
end

Base.rand(d::MEInitStateDist, n::Int=1; kwargs...) = rand(d.rng, d, n; kwargs...)

function normalize_and_weight(lode_map::AbstractArray, mainbody_weight::Real)
    max_lode = maximum(lode_map)
    lode_map ./= max_lode
    lode_map .*= mainbody_weight
    lode_map = repeat(lode_map, outer=(1, 1, 1))
    return lode_map
end

function calc_massive(ore_map::AbstractArray, massive_threshold::Real, dim_scale::Real)
    return dim_scale*sum(ore_map .>= massive_threshold)
end

function extraction_reward(m::MineralExplorationPOMDP, s::MEState)
    truth = size(s.mainbody_map) == m.high_fidelity_dim
    dim_scale = truth ? m.target_dim_scale : m.dim_scale
    r_massive = calc_massive(s.ore_map, m.massive_threshold, dim_scale)
    r = m.strike_reward*r_massive
    r -= m.extraction_cost
    return r
end

function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, b::MEBelief, rng::Random.AbstractRNG)
    error("POMDPs.gen with a belief passed is has not been implemented (yet) — Robert Moss")
end

function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, rng::Random.AbstractRNG)
    if a ∉ POMDPs.actions(m, s)
        error("Invalid Action $a from state $s")
    end
    stopped = s.stopped
    decided = s.decided
    a_type = a.type

    # drill then stop then mine or abandon
    if a_type == :stop && !stopped && !decided
        obs = MEObservation(nothing, true, false)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = false
    elseif a_type == :abandon && stopped && !decided
        obs = MEObservation(nothing, true, true)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
    elseif a_type == :mine && stopped && !decided
        obs = MEObservation(nothing, true, true)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
    elseif a_type ==:drill && !stopped && !decided
        ore_obs = high_fidelity_obs(m, s.ore_map, a)
        a_coords = reshape(Int64[a.coords[1] a.coords[2]], 2, 1)
        rock_obs_p = deepcopy(s.rock_obs)
        rock_obs_p.coordinates = hcat(rock_obs_p.coordinates, a_coords)
        push!(rock_obs_p.ore_quals, ore_obs)
        n_bores = length(rock_obs_p)
        stopped_p = n_bores >= m.max_bores
        decided_p = false
        obs = MEObservation(ore_obs, stopped_p, false)
    else
        error("Invalid Action! Action: $(a.type), Stopped: $stopped, Decided: $decided")
    end
    r = reward(m, s, a)
    sp = MEState(s.ore_map, s.mainbody_params, s.mainbody_map, rock_obs_p, stopped_p, decided_p)
    return (sp=sp, o=obs, r=r)
end


function POMDPs.reward(m::MineralExplorationPOMDP, s::MEState, a::MEAction)
    stopped = s.stopped
    decided = s.decided
    a_type = a.type

    if a_type in [:stop, :abandon]
        r = 0.0
    elseif a_type == :mine
        r = extraction_reward(m, s)
    elseif a_type ==:drill
        r = -m.drill_cost
    else
        error("Invalid Action! Action: $(a.type), Stopped: $stopped, Decided: $decided")
    end
    return r
end


function POMDPs.actions(m::MineralExplorationPOMDP)
    idxs = CartesianIndices(m.grid_dim[1:2])
    bore_actions = reshape(collect(idxs), prod(m.grid_dim[1:2]))
    actions = MEAction[MEAction(type=:stop), MEAction(type=:mine),
                        MEAction(type=:abandon)]
    grid_step = m.grid_spacing + 1
    for coord in bore_actions[1:grid_step:end]
        push!(actions, MEAction(coords=coord))
    end
    return actions
end

function POMDPs.actions(m::MineralExplorationPOMDP, s::MEState)
    if s.decided
        return MEAction[]
    elseif s.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = OrderedSet(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        n_obs = length(s.rock_obs.ore_quals) - n_initial
        if m.max_movement != 0 && n_obs > 0
            d = m.max_movement
            drill_s = s.rock_obs.coordinates[:,end]
            x = drill_s[1]
            y = drill_s[2]
            reachable_coords = CartesianIndices((x-d:x+d,y-d:y+d))
            reachable_acts = MEAction[]
            for coord in reachable_coords
                dx = abs(x - coord[1])
                dy = abs(y - coord[2])
                d2 = sqrt(dx^2 + dy^2)
                if d2 <= d
                    push!(reachable_acts, MEAction(coords=coord))
                end
            end
            push!(reachable_acts, MEAction(type=:stop))
            reachable_acts = OrderedSet(reachable_acts)
            # reachable_acts = OrderedSet([MEAction(coords=coord) for coord in collect(reachable_coords)])
            action_set = intersect(reachable_acts, action_set)
        end
        for i=1:n_obs
            coord = s.rock_obs.coordinates[:, i + n_initial]
            x = Int64(coord[1])
            y = Int64(coord[2])
            keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
            keepout_acts = OrderedSet([MEAction(coords=coord) for coord in keepout])
            setdiff!(action_set, keepout_acts)
        end
        # delete!(action_set, MEAction(type=:mine))
        # delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPModelTools.obs_weight(m::MineralExplorationPOMDP, s::MEState,
                    a::MEAction, sp::MEState, o::MEObservation)
    w = 0.0
    if a.type != :drill
        w = o.ore_quality == nothing ? 1.0 : 0.0
    else
        o_mainbody = high_fidelity_obs(m, s.mainbody_map, a)
        o_gp = (o.ore_quality - o_mainbody)
        mu = m.gp_mean
        sigma = sqrt(m.variogram[1])
        point_dist = Normal(mu, sigma)
        w = pdf(point_dist, o_gp)
    end
    return w
end

function high_fidelity_obs(m::MineralExplorationPOMDP, subsurface_map::Array, a::MEAction)
    if size(subsurface_map) == m.grid_dim
        return subsurface_map[a.coords[1], a.coords[2], 1]
    else
        # truncate drill coordinates to map to high-fidelity grid
        hf_coords = trunc.(Int, a.coords.I ./ m.ratio[1:2])
        return subsurface_map[hf_coords[1], hf_coords[2], 1]
    end
end