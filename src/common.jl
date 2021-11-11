@with_kw mutable struct RockObservations
    ore_quals::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

struct MEState
    ore_map::Array{Float64}  # 3D array of ore_quality values for each grid-cell
    var::Float64 #  Diagonal variance of main ore-body generator
    mainbody_map::Array{Float64}
    rock_obs::RockObservations
    stopped::Bool # Whether or not STOP action has been taken
    particles::Vector{Tuple{Float64, Array{Float64, 3}}} #mainbody belief
end

function Base.length(obs::RockObservations)
    return length(obs.ore_quals)
end

struct MEObservation
    ore_quality::Union{Float64, Nothing}
    stopped::Bool
end

@with_kw struct MEAction
    type::Symbol = :drill
    coords::CartesianIndex = CartesianIndex(0, 0)
end

# struct MEBelief
#     particles::Vector{Tuple{Float64, Array{Float64, 3}}} # Vector of vars & lode maps
#     rock_obs::RockObservations
#     acts::Vector{MEAction}
#     obs::Vector{MEObservation}
#     stopped::Bool
#     full::Bool # Whether or not to generate full state maps # TODO Implement this
#     gp_dist::GeoStatsDistribution
# end

function calc_K(domain, mean, variogram, rock_obs::RockObservations)
    pdomain = domain
    table = DataFrame(ore=rock_obs.ore_quals .- mean)
    domain = PointSet(rock_obs.coordinates)
    pdata = georef(table, domain)
    vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    dlocs = Int64[m[1] for m in vmapping]
    𝒟d = [centroid(pdomain, i) for i in dlocs]
    γ = variogram
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)
    return K
end

function reweight(particles::Vector, rock_obs::RockObservations, grid_dim::Tuple,
                variogram, gp_mean::Float64, gp_weight::Float64, a::MEAction,
                o::MEObservation)
    ws = Float64[]
    bore_coords = rock_obs.coordinates
    n = size(bore_coords)[2]
    ore_obs = [o for o in rock_obs.ore_quals]
    if o.ore_quality != nothing
        bore_coords = hcat(bore_coords, [a.coords[1], a.coords[2]])
        push!(ore_obs, o.ore_quality)
        m = n + 1
    else
        m = n
    end
    domain =  CartesianGrid{Int64}(grid_dim[1], grid_dim[2])
    K = calc_K(domain, gp_mean, variogram, RockObservations(ore_quals=ore_obs, coordinates=bore_coords))
    mu = zeros(Float64, m) .+ gp_mean
    gp_dist = MvNormal(mu, K)
    for (mb_var, mb_map) in particles
        o_n = zeros(Float64, m)
        for i = 1:m
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]]
            o_n[i] = (ore_obs[i] - o_mainbody)/gp_weight
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws .+= 1e-6
    ws ./= sum(ws)
    return ws
end

function resample(particles::Vector, wp::Vector{Float64}, grid_dim::Tuple, mainbody_loc::Vector,
                mainbody_weight::Float64, a::MEAction, o::MEObservation, rng::Random.AbstractRNG)
    n = length(wp)
    sampled_particles = sample(rng, particles, StatsBase.Weights(wp), n, replace=true)
    mainbody_vars = Float64[]
    particles = Tuple{Float64, Array{Float64}}[]
    for (mainbody_var, mainbody_map) in sampled_particles
        if mainbody_var ∈ mainbody_vars
            mainbody_var += randn()
            mainbody_var = clamp(mainbody_var, 0.0, Inf)
            mainbody_map = zeros(Float64, Int(grid_dim[1]), Int(grid_dim[2]))
            cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
            mvnorm = MvNormal(mainbody_loc, cov)
            for i = 1:grid_dim[1]
                for j = 1:grid_dim[2]
                    mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
                end
            end
            max_lode = maximum(mainbody_map)
            mainbody_map ./= max_lode
            mainbody_map .*= mainbody_weight
            mainbody_map = reshape(mainbody_map, grid_dim)
        end
        push!(mainbody_vars, mainbody_var)
        push!(particles, (mainbody_var, mainbody_map))
    end
    return particles
end
