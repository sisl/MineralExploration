struct MEState
    ore_map::Array{Float64}  # 3D array of ore_quality values for each grid-cell
    var::Float64 #  Diagonal variance of main ore-body generator 
    bore_coords::Union{Nothing, Matrix{Int64}} # 2D grid cell location of each well
    stopped::Bool # Whether or not STOP action has been taken
    decided::Bool # Whether or not the extraction decision has been made
end

@with_kw mutable struct RockObservations
    ore_quals::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
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
