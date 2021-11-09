
struct MEBelief
    particles::Vector{Tuple{Float64, Array{Float64, 3}}} # Vector of vars & lode maps
    rock_obs::RockObservations
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
    stopped::Bool
    decided::Bool
    full::Bool # Whether or not to generate full state maps # TODO Implement this
    gp_dist::GeoStatsDistribution
end

struct MEBeliefUpdater <: POMDPs.Updater
    m::MineralExplorationPOMDP
    n::Int64
    full::Bool
    rng::AbstractRNG
end

MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int64, full::Bool=false) =
                                MEBeliefUpdater(m, n, full, Random.GLOBAL_RNG)

function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    x_dim = d.gp_distribution.grid_dims[1]
    y_dim = d.gp_distribution.grid_dims[2]

    particles = Tuple{Float64, Array{Float64}}[]
    for i=1:up.n
        lode_map = zeros(Float64, x_dim, y_dim)
        mainbody_var = rand(up.rng)*(up.m.mainbody_var_max - up.m.mainbody_var_min) + up.m.mainbody_var_min
        cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
        mvnorm = MvNormal(d.mainbody_loc, cov)
        for i = 1:x_dim
            for j = 1:y_dim
                lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
            end
        end
        max_lode = maximum(lode_map)
        lode_map ./= max_lode
        lode_map .*= d.mainbody_weight
        lode_map = repeat(lode_map, outer=(1, 1, 1))

        push!(particles, (mainbody_var, lode_map))
    end

    gp_dist = GeoStatsDistribution(up.m)
    rock_obs = RockObservations()
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(particles, rock_obs, acts, obs, false, false, up.full, gp_dist)
end

function calc_K(b::MEBelief, rock_obs::RockObservations)
    pdomain = b.gp_dist.domain
    table = DataFrame(ore=rock_obs.ore_quals .- b.gp_dist.mean)
    domain = PointSet(rock_obs.coordinates)
    pdata = georef(table, domain)
    vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    dlocs = Int64[m[1] for m in vmapping]
    𝒟d = [centroid(pdomain, i) for i in dlocs]
    γ = b.gp_dist.variogram
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)
    return K
end

function reweight(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)
    ws = Float64[]
    bore_coords = b.rock_obs.coordinates
    n = size(bore_coords)[2]
    bore_coords = hcat(bore_coords, [a.coords[1], a.coords[2]])
    ore_obs = [o.ore_quality for o in b.obs]
    push!(ore_obs, o.ore_quality)
    K = calc_K(b, RockObservations(ore_quals=ore_obs, coordinates=bore_coords))
    mu = zeros(Float64, n+1) .+ up.m.gp_mean
    gp_dist = MvNormal(mu, K)
    for (mb_var, mb_map) in b.particles
        o_n = zeros(Float64, n+1)
        for i = 1:n+1
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]]
            o_n[i] = (ore_obs[i] - o_mainbody)/up.m.gp_weight
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws ./= sum(ws) + 1e-6
    return ws
end

function resample(up::MEBeliefUpdater, b::MEBelief, wp::Vector{Float64}, a::MEAction, o::MEObservation)
    sampled_particles = sample(up.rng, b.particles, StatsBase.Weights(wp), up.n, replace=true)
    mainbody_vars = Float64[]
    particles = Tuple{Float64, Array{Float64}}[]
    for (mainbody_var, mainbody_map) in sampled_particles
        if mainbody_var ∈ mainbody_vars
            mainbody_var += randn()
            mainbody_var = clamp(mainbody_var, 0.0, Inf)
            mainbody_map = zeros(Float64, Int(up.m.grid_dim[1]), Int(up.m.grid_dim[2]))
            cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
            mvnorm = MvNormal(up.m.mainbody_loc, cov)
            for i = 1:up.m.grid_dim[1]
                for j = 1:up.m.grid_dim[2]
                    mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
                end
            end
            max_lode = maximum(mainbody_map)
            mainbody_map ./= max_lode
            mainbody_map .*= up.m.mainbody_weight
            mainbody_map = reshape(mainbody_map, up.m.grid_dim)
        end
        push!(mainbody_vars, mainbody_var)
        push!(particles, (mainbody_var, mainbody_map))
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)
    wp = reweight(up, b, a, o)
    pp = resample(up, b, wp, a, o)
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
                            a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_particles = Tuple{Float64, Array{Float64}}[p for p in b.particles]
        bp_rock = b.rock_obs
    else
        bp_particles = update_particles(up, b, a, o)
        bp_rock = b.rock_obs
        bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]])
        push!(bp_rock.ore_quals, o.ore_quality)
    end

    bp_acts = MEAction[]
    for act in b.acts
        push!(bp_acts, act)
    end
    push!(bp_acts, a)

    bp_obs = MEObservation[]
    for obs in b.obs
        push!(bp_obs, obs)
    end
    push!(bp_obs, o)

    bp_stopped = o.stopped
    bp_decided = o.decided
    bp_full = b.full

    bp_geostats = GeoStatsDistribution(b.gp_dist.grid_dims, bp_rock,
                                        b.gp_dist.domain, b.gp_dist.mean,
                                        b.gp_dist.variogram)
    return MEBelief(bp_particles, bp_rock, bp_acts, bp_obs, bp_stopped,
                    bp_decided, bp_full, bp_geostats)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    mainbody_var, lode_map = rand(rng, b.particles)
    if b.full
        gp_ore_map = Base.rand(rng, b.geostats)
        gp_ore_map .*= b.geostats.gp_weight
        ore_map = lode_map + gp_ore_map
        clamp!(ore_map, 0.0, Inf)
    else
        ore_map = zero(lode_map) .- 1.0
    end
    return MEState(ore_map, mainbody_var, lode_map, deepcopy(b.rock_obs),
                    b.stopped, false)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    (x, y, z) = size(b.particles[1][2])
    μ = zeros(Float64, x, y, z)
    w = 1.0/length(b.particles)
    for p in b.particles
        ore_map = p[2]
        μ .+= ore_map .* w
    end
    σ² = zeros(Float64, x, y, z)
    for p in b.particles
        ore_map = p[2]
        σ² .+= w*(ore_map - μ).^2
    end
    return (μ, σ²)
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::MEBelief)
    if b.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(b.rock_obs.ore_quals)
            n_obs = length(b.rock_obs.ore_quals) - n_initial
            for i=1:n_obs
                coord = b.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
                setdiff!(action_set, keepout_acts)
            end
        end
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::POMCPOW.StateBelief)
    o = b.sr_belief.o
    s = rand(b.sr_belief.dist)[1]
    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(s.rock_obs.ore_quals)
            n_obs = length(s.rock_obs.ore_quals) - n_initial
            for i=1:n_obs
                coord = s.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
                setdiff!(action_set, keepout_acts)
            end
        end
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)
    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = Set(POMDPs.actions(m))
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function mean_var(b::MEBelief)
    vars = [s[1] for s in b.particles]
    mean(vars)
end

function std_var(b::MEBelief)
    vars = [s[1] for s in b.particles]
    std(vars)
end

function Plots.plot(b::MEBelief, t=nothing)
    mean, var = summarize(b)
    if t == nothing
        mean_title = "Belief Mean"
        std_title = "Belief StdDev"
    else
        mean_title = "Belief Mean t=$t"
        std_title = "Belief StdDev t=$t"
    end
    fig1 = heatmap(mean[:,:,1], title=mean_title, fill=true, clims=(0.0, 1.0), legend=:none)
    fig2 = heatmap(sqrt.(var[:,:,1]), title=std_title, fill=true, legend=:none, clims=(0.0, 0.2))
    if !isempty(b.rock_obs.ore_quals)
        x = b.rock_obs.coordinates[2, :]
        y = b.rock_obs.coordinates[1, :]
        plot!(fig1, x, y, seriestype = :scatter)
        plot!(fig2, x, y, seriestype = :scatter)
    end
    fig = plot(fig1, fig2, layout=(1,2))
    return fig
end
