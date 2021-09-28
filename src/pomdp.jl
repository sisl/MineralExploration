
@with_kw struct MineralExplorationPOMDP <: POMDP{MEState, CartesianIndex, MEObservation} end
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (80, 80, 1) #  dim x dim grid size
    max_bores::Int64 = 3 # Maximum number of bores
    time_interval::Float64 = 1.0 # Minimum time between bores (in months)
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 2 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 1 # Number of cells in between each cell in which wells can be placed
    variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    nugget::Tuple = (1, 0)
    gp_weight::Float64 = 0.5
    mainbody_weight::Float64 = 0.5
    mainbody_var::Matrix{Float64} = [100.0 0.0; 0.0 100.0]
end

function GSLIBDistribution(p::MinearalExplorationPOMDP)
    return GSLIBDistribution(grid_dims=p.grid_dim, n=p.grid_dim,
            data=p.initial_data, variogram=p.poro_variogram, nugget=p.poro_nugget)
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

function sample_initial(p::MinearalExplorationPOMDP, n::Integer)
    coords, coords_array = sample_coords(p.grid_dim, n)
    dist = GSLIBDistribution(p)
    state_tuple = rand(dist)
    porosity = state_tuple[1][coords]
    return RockObservations(porosity, coords_array)
end

function sample_initial(p::MinearalExplorationPOMDP, coords::Array)
    n = length(coords)
    coords_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(coords)
        coords_array[1, i] = sample[1]
        coords_array[2, i] = sample[2]
    end
    dist = GSLIBDistribution(p)
    state_tuple = rand(dist)
    porosity = state_tuple[1][coords]
    return RockObservations(porosity, coords_array)
end

function initialize_data!(p::MinearalExplorationPOMDP, n::Integer)
    new_rock_obs = sample_initial(p, n)
    append!(p.initial_data.porosity, new_rock_obs.porosity)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

function initialize_data!(p::MinearalExplorationPOMDP, coords::Array)
    new_rock_obs = sample_initial(p, coords)
    append!(p.initial_data.porosity, new_rock_obs.porosity)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

### DO NOT OVERWRITE ###
POMDPs.discount(::MineralExplorationPOMDP) = 0.95
POMDPs.isterminal(m::MineralExplorationPOMDP, s::MEState) = size(s.t)[2] >= m.max_bores

struct MEInitStateDist
    gp_distribution::GSLIBDistribution
    rng::AbstractRNG
end

function POMDPs.initialstate_distribution(m::MineralExplorationPOMDP)
    reservoir_dist = GSLIBDistribution(m.spec)
    CCSInitStateDist(reservoir_dist, m.rng)
end

function Base.rand(d::CCSInitStateDist)
    rock_state = Base.rand(d.rng, d.reservoir_distribution)
    MEState(rock_state[1], rock_state[2], rock_state[3],
            d.reservoir_distribution.data.coordinates, nothing)
end

function gen_reward(s::MEState, a::CartesianIndex, sp::MEState)
    return 0.0
end # TODO


### Implement for POMDP types
function POMDPs.gen(m::MineralExplorationPOMDP, s, a, rng)
    throw("POMDPs.gen function not implemented for $(typeof(m)) type")
    return (sp=sp, o=o, r=r)
end

function POMDPs.actions(m::MineralExplorationPOMDP)
    throw("POMDPs.actions function not implemented for $(typeof(m)) type")
    return actions
end

function POMDPs.actions(m::MineralExplorationPOMDP, s::MEState)
    throw("POMDPs.actions function not implemented for $(typeof(m)) type")
    return actions
end

function POMDPs.actions(m::MineralExplorationPOMDP, b)
    throw("POMDPs.actions function not implemented for $(typeof(m)) type")
    return actions
end

# For POMCPOW
function POMDPModelTools.obs_weight(p::MineralExplorationPOMDP, s, a, sp, o)
    throw("POMDPModelTools.obs_weight function not implemented for $(typeof(m)) type")
    return weight
end
