
@with_kw struct MineralExplorationPOMDP <: POMDP{MEState, MEAction, MEObservation}
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) #  dim x dim grid size
    max_bores::Int64 = 10 # Maximum number of bores
    time_interval::Float64 = 1.0 # Minimum time between bores (in months)
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 1 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 1 # Number of cells in between each cell in which wells can be placed
    drill_cost::Float64 = 0.1
    strike_reward::Float64 = 1.0
    extraction_cost::Float64 = 150.0
    extraction_lcb::Float64 = 0.9
    variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    nugget::Tuple = (1, 0)
    gp_weight::Float64 = 0.35
    mainbody_weight::Float64 = 0.45
    mainbody_loc::Vector{Float64} = [25.0, 25.0]
    mainbody_var_min::Float64 = 40.0
    mainbody_var_max::Float64 = 80.0
    massive_threshold::Float64 = 0.7
    rng::AbstractRNG = Random.GLOBAL_RNG
end

function GSLIBDistribution(p::MineralExplorationPOMDP)
    return GSLIBDistribution(grid_dims=p.grid_dim, n=p.grid_dim,
            data=p.initial_data, variogram=p.variogram, nugget=p.nugget)
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
    dist = GSLIBDistribution(p)
    state = rand(dist, silent=true)
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
    dist = GSLIBDistribution(p)
    state = rand(dist)
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

struct MEInitStateDist
    gp_distribution::GSLIBDistribution
    gp_weight::Float64
    mainbody_weight::Float64
    mainbody_loc::Vector{Float64}
    mainbody_var_max::Float64
    mainbody_var_min::Float64
    massive_thresh::Float64
    rng::AbstractRNG
end

function POMDPs.initialstate_distribution(m::MineralExplorationPOMDP)
    reservoir_dist = GSLIBDistribution(m)
    MEInitStateDist(reservoir_dist, m.gp_weight, m.mainbody_weight,
                    m.mainbody_loc, m.mainbody_var_max, m.mainbody_var_min,
                    m.massive_threshold, m.rng)
end

function Base.rand(d::MEInitStateDist)
    gp_ore_map = Base.rand(d.rng, d.gp_distribution)
    mean_gp = mean(gp_ore_map)
    gp_ore_map ./= mean_gp
    gp_ore_map .*= d.gp_weight

    clamp!(gp_ore_map, 0.0, d.massive_thresh)

    x_dim = d.gp_distribution.grid_dims[1]
    y_dim = d.gp_distribution.grid_dims[2]
    lode_map = zeros(x_dim, y_dim)
    mainbody_var = rand(d.rng)*(d.mainbody_var_max - d.mainbody_var_min) + d.mainbody_var_min
    cov = [mainbody_var 0.0; 0.0 mainbody_var]
    mvnorm = MvNormal(d.mainbody_loc, cov)
    for i = 1:x_dim
        for j = 1:y_dim
            lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    max_lode = maximum(lode_map)
    lode_map ./= max_lode
    lode_map .*= d.mainbody_weight
    lode_map = repeat(lode_map, outer=(1, 1, 8))

    ore_map = lode_map + gp_ore_map
    clamp!(ore_map, 0.0, 1.0)
    MEState(ore_map, mainbody_var,
            d.gp_distribution.data.coordinates, false, false)
end

Base.rand(rng::AbstractRNG, d::MEInitStateDist) = rand(d)

function extraction_reward(m::MineralExplorationPOMDP, s::MEState)
    r = m.strike_reward*sum(s.ore_map[:,:,1] .>= m.massive_threshold)
    r -= m.extraction_cost
    return r
end

function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, rng::Random.AbstractRNG)
    stopped = s.stopped
    decided = s.decided
    a_type = a.type
    if a_type == :stop && !stopped && !decided
        obs = MEObservation(nothing, true, false)
        r = 0.0
        coords_p = s.bore_coords
        stopped_p = true
        decided_p = false
    elseif a_type == :abandon && stopped && !decided
        obs = MEObservation(nothing, true, true)
        r = 0.0
        coords_p = s.bore_coords
        stopped_p = true
        decided_p = true
    elseif a_type == :mine && stopped && !decided
        obs = MEObservation(nothing, true, true)
        r = extraction_reward(m, s)
        coords_p = s.bore_coords
        stopped_p = true
        decided_p = true
    elseif a_type ==:drill && !stopped && !decided
        ore_obs = s.ore_map[a.coords[1], a.coords[2], 1]
        a = reshape(Int64[a.coords[1] a.coords[2]], 2, 1)
        r = -m.drill_cost
        coords_p = [s.bore_coords a]
        n_bores = size(coords_p)[2]
        stopped_p = n_bores >= m.max_bores
        decided_p = false
        obs = MEObservation(ore_obs, stopped_p, false)
    else
        error("Invalid Action! Action: $a, Stopped: $stopped, Decided: $decided")
    end
    sp = MEState(s.ore_map, s.var, coords_p, stopped_p, decided_p)
    return (sp=sp, o=obs, r=r)
end


function POMDPs.actions(m::MineralExplorationPOMDP)
    idxs = CartesianIndices(m.grid_dim[1:2])
    bore_actions = reshape(collect(idxs), prod(m.grid_dim[1:2]))
    actions = MEAction[MEAction(type=:stop), MEAction(type=:mine),
                        MEAction(type=:abandon)]
    for coord in bore_actions
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
        action_set = Set(POMDPs.actions(m))
        for i=1:size(s.bore_coords)[2]
            coord = s.bore_coords[:, i]
            x = Int64(coord[1])
            y = Int64(coord[2])
            keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
            keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
            setdiff!(action_set, keepout_acts)
        end
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
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
        ore = s.ore_map[:,:,1][a.coords]
        dist = Normal(ore, m.obs_noise_std)
        w = pdf(dist, o.ore_quality)
    end
    return w
end
