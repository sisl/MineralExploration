
struct MEBelief
    bore_coords::Union{Nothing, Matrix{Int64}}
    stopped::Bool
    particles::Vector{Float64}
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
end

struct MEBeliefUpdater <: POMDPs.Updater
    m::MineralExplorationPOMDP
    n::Int64
    vars::Vector{Float64}
    rng::AbstractRNG
end

function MEBeliefUpdater(m::MineralExplorationPOMDP, n_particles::Int64, n_vars::Int64=40,
                        rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    vars = LinRange(m.mainbody_var_min, m.mainbody_var_max, n_vars)
    return MEBeliefUpdater(m, n_particles, vars, rng)
end

function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    x_dim = d.gp_distribution.grid_dims[1]
    y_dim = d.gp_distribution.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    mainbody_var = (d.mainbody_var_max + d.mainbody_var_min)/2.0
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
    lode_map = repeat(lode_map, outer=(1, 1, 8))

    gp_ore_maps = Base.rand(d.rng, d.gp_distribution, up.n)
    particles = MEState[]
    for gp_ore_map in gp_ore_maps
        gp_ore_map ./= 0.3 # TODO
        gp_ore_map .*= d.gp_weight
        clamp!(gp_ore_map, 0.0, d.massive_thresh)
        ore_map = lode_map + gp_ore_map
        clamp!(ore_map, 0.0, 1.0)
        s = MEState(ore_map, mainbody_var, d.gp_distribution.data.coordinates,
                    false, false)
        push!(particles, s)
    end
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(nothing, false, particles, acts, obs)
end

function variogram(x, y, a, c) # point 1, point 2, range, sill
    h = sqrt(sum((x - y).^2))
    if h <= a
        return c*(1.5*h/a - 0.5*(h/a)^3)
    else
        return c
    end
end

function calc_var(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)
    bore_coords = b.particles[1].bore_coords
    n = size(bore_coords)[2]
    bore_coords = hcat(bore_coords, [a.coords[1], a.coords[2]])
    ore_obs = [o.ore_quality for o in b.obs]
    push!(ore_obs, o.ore_quality)
    K = zeros(Float64, n+1, n+1)
    marginal_var = 0.005 # TODO
    for i = 1:n+1
        for j = 1:n+1
            K[i, j] = clamp(marginal_var - variogram(bore_coords[:, i], bore_coords[:, j], 30.0, marginal_var), 0.0, Inf) # TODO
            if i == j
                K[i, j] += 1e-4 # TODO
            end
        end
    end

    mu = zeros(Float64, n+1) .+ 0.3 # TODO
    gp_dist = MvNormal(mu, K)

    map_var = 0.0
    w_total = 0.0
    for mb_var in up.vars
        mainbody_cov = Distributions.PDiagMat([mb_var, mb_var])
        mainbody_dist = MvNormal(up.m.mainbody_loc, mainbody_cov)
        mainbody_max = 1.0/(2*π*mb_var)
        o_n = zeros(Float64, n+1)
        for i = 1:n+1
            o_mainbody = pdf(mainbody_dist, bore_coords[:, i])
            o_mainbody /= mainbody_max
            o_mainbody *= up.m.mainbody_weight
            o_n[i] = (ore_obs[i] - o_mainbody)*0.3/up.m.gp_weight
        end
        w = pdf(gp_dist, o_n)
        map_var += w*mb_var
        w_total += w
    end
    return map_var/w_total
end

function resample(up::MEBeliefUpdater, b::MEBelief, mb_var::Float64, a::MEAction, o::MEObservation)
    x_dim = up.m.grid_dim[1]
    y_dim = up.m.grid_dim[2]
    mainbody_map = zeros(Float64, x_dim, y_dim)
    cov = Distributions.PDiagMat([mb_var, mb_var])
    mvnorm = MvNormal(up.m.mainbody_loc, cov)
    for i = 1:x_dim
        for j = 1:y_dim
            mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    max_lode = maximum(mainbody_map)
    mainbody_map ./= max_lode
    mainbody_map .*= up.m.mainbody_weight

    ore_quals = Float64[o.ore_quality for o in b.obs]
    push!(ore_quals, o.ore_quality)
    if b.bore_coords isa Nothing
        ore_coords = zeros(Int64, 2, 0)
    else
        ore_coords = b.bore_coords
    end
    ore_coords = hcat(ore_coords, [a.coords[1], a.coords[2]])
    geostats_dist = GeoStatsDistribution(up.m)
    geostats_dist.data.coordinates = ore_coords
    n_ore_quals = Float64[]
    for (i, ore_qual) in enumerate(ore_quals)
        prior_ore = mainbody_map[ore_coords[1, i], ore_coords[2, i]]
        n_ore_qual = (ore_qual - prior_ore).*(0.3/up.m.gp_weight)
        push!(n_ore_quals, n_ore_qual)
    end
    geostats_dist.data.ore_quals = n_ore_quals

    mainbody_map_3d = repeat(mainbody_map, outer=(1, 1, 8))
    particles = MEState[]
    gp_ore_maps = Base.rand(up.rng, geostats_dist, up.n)
    for gp_ore_map in gp_ore_maps
        gp_ore_map ./= 0.3
        gp_ore_map .*= up.m.gp_weight
        clamp!(gp_ore_map, 0.0, up.m.massive_threshold)
        ore_map = gp_ore_map .+ mainbody_map_3d
        clamp!(ore_map, 0.0, 1.0)

        new_state = MEState(ore_map, mb_var,
                geostats_dist.data.coordinates, false, false)
        push!(particles, new_state)
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)
    # wp = reweight(up, b, b.particles, a, o)
    mb_var = calc_var(up, b, a, o)
    pp = resample(up, b, mb_var, a, o)
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
                            a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_coords = b.bore_coords
        bp_particles = MEState[]
        for p in b.particles
            s = MEState(p.ore_map, p.var, p.bore_coords, o.stopped, o.decided)
            push!(bp_particles, s)
        end
    else
        if b.bore_coords == nothing
            bp_coords = reshape([a.coords[1], a.coords[2]], 2, 1)
        else
            bp_coords = hcat(b.bore_coords, [a.coords[1], a.coords[2]])
        end
        bp_particles = update_particles(up, b, a, o)
    end
    bp_stopped = o.stopped
    bp_decided = o.decided

    bp_acts = MEAction[]
    bp_obs = MEObservation[]
    for act in b.acts
        push!(bp_acts, act)
    end
    push!(bp_acts, a)
    for obs in b.obs
        push!(bp_obs, obs)
    end
    push!(bp_obs, o)
    return MEBelief(bp_coords, bp_stopped, bp_particles, bp_acts, bp_obs)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    s0 = rand(rng, b.particles)
    return MEState(deepcopy(s0.ore_map), s0.var, deepcopy(s0.bore_coords), b.stopped, false)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    (x, y, z) = size(b.particles[1].ore_map)
    μ = zeros(Float64, x, y, z)
    w = 1.0/length(b.particles)
    for (i, p) in enumerate(b.particles)
        ore_map = convert(Array{Float64, 3}, p.ore_map)
        μ .+= ore_map .* w
    end
    σ² = zeros(Float64, x, y, z)
    for (i, p) in enumerate(b.particles)
        ore_map = convert(Array{Float64, 3}, p.ore_map)
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
        if b.bore_coords != nothing
            n_obs = size(b.bore_coords)[2] - n_initial
            for i=1:n_obs
                coord = b.bore_coords[:, i + n_initial]
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
        if s.bore_coords != nothing
            n_obs = size(s.bore_coords)[2] - n_initial
            for i=1:n_obs
                coord = s.bore_coords[:, i + n_initial]
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
    vars = [s.var for s in b.particles]
    mean(vars)
end

function std_var(b::MEBelief)
    vars = [s.var for s in b.particles]
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
    if b.bore_coords != nothing
        x = b.bore_coords[2, :]
        y = b.bore_coords[1, :]
        plot!(fig1, x, y, seriestype = :scatter)
        plot!(fig2, x, y, seriestype = :scatter)
    end
    fig = plot(fig1, fig2, layout=(1,2))
    return fig
end
