
@with_kw struct MineralExplorationPOMDP <: POMDP{MEState, Union{CartesianIndex, Symbol}, MEObservation}
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) #  dim x dim grid size
    max_bores::Int64 = 10 # Maximum number of bores
    time_interval::Float64 = 1.0 # Minimum time between bores (in months)
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 1 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 1 # Number of cells in between each cell in which wells can be placed
    obs_noise_std::Float64 = 0.01
    drill_cost::Float64 = 0.1
    strike_reward::Float64 = 1.0
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
    state = rand(dist, silent=false)
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
POMDPs.isterminal(m::MineralExplorationPOMDP, s::MEState) = size(s.bore_coords)[2] >= m.max_bores ||
                                                                stopped

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
    MEState(ore_map,
            d.gp_distribution.data.coordinates, false)
end

Base.rand(rng::AbstractRNG, d::MEInitStateDist) = rand(d)

function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a, rng)
    if a == :stop
        obs = MEObservation(nothing)
        r = 0.0
        coords_p = s.bore_coords
        stopped = true
    else
        ore_obs = s.ore_map[a[1], a[2], 1]
        obs = MEObservation(ore_obs)
        r = obs >= m.massive_threshold ? m.strike_reward : 0.0
        r -= m.drill_cost
        a = reshape(Int64[a[1] a[2]], 2, 1)
        coords_p = [s.bore_coords a]
    end
    sp = MEState(s.ore_map, coords_p, stopped)
    return (sp=sp, o=obs, r=r)
end

function POMDPs.actions(m::MineralExplorationPOMDP)
    idxs = CartesianIndices(m.grid_dim[1:2])
    bore_actions = reshape(collect(idxs), prod(m.grid_dim[1:2]))
    actions = Union{CartesianIndex{2}, Symbol}[:stop]
    append!(actions, bore_actions)
    return actions
end

function POMDPs.actions(m::MineralExplorationPOMDP, s::MEState)
    action_set = Set(POMDPs.actions(m))
    for i=1:size(s.bore_coords)[2]
        coord = s.bore_coords[:, i]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

function POMDPs.actions(m::MineralExplorationPOMDP, b)
    action_set = Set(POMDPs.actions(m))
    n_initial = length(m.initial_data)
    n_obs = size(b.rock_belief.data.coordinates)[2] - n_initial
    for i=1:n_obs
        coord = b.rock_belief.data.coordinates[:, i + n_initial]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

# For POMCPOW
function POMDPModelTools.obs_weight(p::MineralExplorationPOMDP, s, a, sp, o)
    throw("POMDPModelTools.obs_weight function not implemented for $(typeof(m)) type")
    return weight
end
