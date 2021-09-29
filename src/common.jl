struct MEState
    ore_map::Array{Float64}  # 3D array of ore_quality values for each grid-cell
    bore_coords::Union{Nothing, Matrix{Int64}} # 2D grid cell location of each well
    stopped::Bool # Whether or not STOP action has been taken
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
end
