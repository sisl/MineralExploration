
struct MEBelief{G}
    particles::Vector{MEState} # Vector of vars & lode maps
    rock_obs::RockObservations
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
    stopped::Bool
    decided::Bool
    geostats::G #GSLIB or GeoStats
end

struct MEBeliefUpdater{G} <: POMDPs.Updater
    m::MineralExplorationPOMDP
    geostats::G
    n::Int64
    noise::Float64
    updates::Int64
    rng::AbstractRNG
end

MEBeliefUpdater(m::MineralExplorationPOMDP, geostats::GeoDist, n::Int64,
                noise::Float64=1.0, updates::Int64=10) = MEBeliefUpdater(m, geostats, n, noise, updates, Random.GLOBAL_RNG)

function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    particles = rand(d, up.n)
    init_rocks = up.m.initial_data
    rock_obs = RockObservations(init_rocks.ore_quals, init_rocks.coordinates)
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(particles, rock_obs, acts, obs, false, false, up.geostats)
end

function calc_K(geostats::GeoDist, rock_obs::RockObservations)
    pdomain = geostats.domain
    table = DataFrame(ore=rock_obs.ore_quals .- geostats.mean)
    domain = PointSet(rock_obs.coordinates)
    pdata = georef(table, domain)
    vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    dlocs = Int64[m[1] for m in vmapping]
    𝒟d = [centroid(pdomain, i) for i in dlocs]
    γ = geostats.variogram
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)
    return K
end

function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)
    ws = Float64[]
    bore_coords = rock_obs.coordinates
    n = size(bore_coords)[2]
    ore_obs = [o for o in rock_obs.ore_quals]
    K = calc_K(geostats, rock_obs)
    mu = zeros(Float64, n) .+ up.m.gp_mean
    gp_dist = MvNormal(mu, K)
    for s in particles
        mb_var = s.var
        mb_map = s.mainbody_map
        o_n = zeros(Float64, n)
        for i = 1:n
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]]
            o_n[i] = (ore_obs[i] - o_mainbody)/up.m.gp_weight
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws .+= 1e-9
    ws ./= sum(ws)
    return ws
end

function resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64},
                geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    sampled_particles = sample(up.rng, particles, StatsBase.Weights(wp), up.n, replace=true)
    mainbody_vars = Float64[]
    mainbody_maps = Array{Float64, 3}[]
    particles = MEState[]
    x = nothing
    ore_quals = deepcopy(rock_obs.ore_quals)
    for s in sampled_particles
        mainbody_var = s.var
        mainbody_map = s.mainbody_map
        ore_map = s.ore_map
        if mainbody_var ∈ mainbody_vars
            mainbody_var += 2.0*(rand() - 0.5)*up.noise
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
            # clamp!(ore_map, 0.0, 1.0)
        end
        n_ore_quals = Float64[]
        for (i, ore_qual) in enumerate(ore_quals)
            prior_ore = mainbody_map[rock_obs.coordinates[1, i], rock_obs.coordinates[2, i], 1]
            n_ore_qual = (ore_qual - prior_ore)./up.m.gp_weight
            push!(n_ore_quals, n_ore_qual)
        end
        geostats.data.ore_quals = n_ore_quals
        # gslib_dist.data.ore_quals = n_ore_quals
        gp_ore_map = Base.rand(up.rng, geostats)
        ore_map = gp_ore_map.*up.m.gp_weight .+ mainbody_map
        rock_obs_p = RockObservations(rock_obs.ore_quals, rock_obs.coordinates)
        sp = MEState(ore_map, mainbody_var, mainbody_map, rock_obs_p,
                    o.stopped, o.decided)
        push!(mainbody_vars, mainbody_var)
        push!(particles, sp)
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, particles::Vector{MEState},
                        geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    pp = particles
    pp_new = nothing
    for i = 1:up.updates
        wp = reweight(up, geostats, pp, rock_obs)
        pp = resample(up, pp, wp, geostats, rock_obs, a, o)
        pp_new = pp
    end
    return pp_new
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
                            a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_particles = MEState[] # MEState[p for p in b.particles]
        for p in b.particles
            s = MEState(p.ore_map, p.var, p.mainbody_map, p.rock_obs, o.stopped, o.decided)
            push!(bp_particles, s)
        end
        bp_rock = RockObservations(ore_quals=deepcopy(b.rock_obs.ore_quals),
                                coordinates=deepcopy(b.rock_obs.coordinates))
        # TODO Swap GeoStatsDistribtuion with genera
        bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, bp_rock,
                                        b.geostats.domain, b.geostats.mean,
                                        b.geostats.variogram, b.geostats.lu_params)
    else
        bp_rock = deepcopy(b.rock_obs)
        bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]])
        push!(bp_rock.ore_quals, o.ore_quality)
        bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock),
                                        b.geostats.domain, b.geostats.mean,
                                        b.geostats.variogram, b.geostats.lu_params)
        update!(bp_geostats, bp_rock)
        bp_particles = update_particles(up, b.particles, bp_geostats, bp_rock, a, o)
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

    return MEBelief(bp_particles, bp_rock, bp_acts, bp_obs, bp_stopped,
                    bp_decided, bp_geostats)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    return rand(rng, b.particles)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    (x, y, z) = size(b.particles[1].ore_map)
    μ = zeros(Float64, x, y, z)
    w = 1.0/length(b.particles)
    for p in b.particles
        ore_map = p.ore_map
        μ .+= ore_map .* w
    end
    σ² = zeros(Float64, x, y, z)
    for p in b.particles
        ore_map = p.ore_map
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
            if n_obs < m.min_bores
                delete!(action_set, MEAction(type=:stop))
            end
        elseif m.min_bores > 0
            delete!(action_set, MEAction(type=:stop))
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
            if n_obs < m.min_bores
                delete!(action_set, MEAction(type=:stop))
            end
        elseif m.min_bores > 0
            delete!(action_set, MEAction(type=:stop))
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
    fig1 = heatmap(mean[:,:,1], title=mean_title, fill=true) #, clims=(0.0, 1.0)) , legend=:none)
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
