@with_kw mutable struct RockObservations
    ore_quals::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

struct MEState{MB}
    ore_map::Array{Float64}  # 3D array of ore_quality values for each grid-cell
    mainbody_params::MB #  Diagonal variance of main ore-body generator
    mainbody_map::Array{Float64}
    rock_obs::RockObservations
    stopped::Bool # Whether or not STOP action has been taken
    decided::Bool # Whether or not the extraction decision has been made
end

function Base.length(obs::RockObservations)
    return length(obs.ore_quals)
end

struct MEObservation
    ore_quality::Union{Float64, Nothing}
    stopped::Bool
    decided::Bool
end

@with_kw struct MEAction
    type::Symbol = :drill
    coords::CartesianIndex = CartesianIndex(0, 0)
end

abstract type GeoDist end

abstract type MainbodyGen end

@with_kw struct MineralExplorationPOMDP <: POMDP{MEState, MEAction, MEObservation}
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) #  dim x dim grid size
    high_fidelity_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) # grid dimensions for high-fidelity case (the "truth" uses this)
    target_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) # grid dimension as the "intended" high-fidelity (i.e., the standard grid dimension that was used to select `extraction_cost` etc.)
    ratio::Tuple{Float64, Float64, Float64} = grid_dim ./ target_dim # scaling "belief" ratio relative to default grid dimensions of 50x50
    target_ratio::Tuple{Float64, Float64, Float64} = high_fidelity_dim ./ target_dim # scaling "truth" ratio relative to default grid dimensions of 50x50
    dim_scale::Float64 = 1/prod(ratio) # scale ore value per cell (for "belief")
    target_dim_scale::Float64 = 1/prod(target_ratio) # scale ore value per cell (for "truth")
    max_bores::Int64 = 10 # Maximum number of bores
    min_bores::Int64 = 1 # Minimum number of bores
    original_max_movement::Int64 = 0 # Original maximum distanace between bores in the default 50x50 grid. 0 denotes no restrictions
    max_movement::Int64 = round(Int, original_max_movement*ratio[1]) # Maximum distanace between bores (scaled based on the ratio). 0 denotes no restrictions
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 1 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 0 # Number of cells in between each cell in which wells can be placed
    drill_cost::Float64 = 0.1
    strike_reward::Float64 = 1.0
    extraction_cost::Float64 = 150.0
    extraction_lcb::Float64 = 0.1
    extraction_ucb::Float64 = 0.1
    variogram::Tuple = (0.005, 30.0, 0.0001) #sill, range, nugget
    # nugget::Tuple = (1, 0)
    geodist_type::Type = GeoStatsDistribution # GeoDist type for geo noise
    gp_mean::Float64 = 0.25
    mainbody_weight::Float64 = 0.6
    true_mainbody_gen::MainbodyGen = BlobNode(grid_dims=high_fidelity_dim) # high-fidelity true mainbody generator
    mainbody_gen::MainbodyGen = BlobNode(grid_dims=grid_dim)
    massive_threshold::Float64 = 0.7
    target_mass_params::Tuple{Real, Real} = (extraction_cost, extraction_cost/3) # target mean and std when standardizing ore mass distributions
    rng::AbstractRNG = Random.GLOBAL_RNG
    c_exp::Float64 = 1.0
end

struct MEInitStateDist
    true_gp_distribution::GeoDist
    gp_distribution::GeoDist
    mainbody_weight::Float64
    true_mainbody_gen::MainbodyGen
    mainbody_gen::MainbodyGen
    massive_thresh::Float64
    dim_scale::Float64
    target_dim_scale::Float64
    target_μ::Float64
    target_σ::Float64
    rng::AbstractRNG
end
