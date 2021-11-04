
struct MEBelief
    bore_coords::Union{Nothing, Matrix{Int64}}
    stopped::Bool
    particles::Vector{MEState}
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
end

struct MEBeliefUpdater <: POMDPs.Updater
    m::MineralExplorationPOMDP
    n::Int64
    rng::AbstractRNG
end

MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int;
    rng::AbstractRNG=Random.GLOBAL_RNG) = MEBeliefUpdater(m, n, rng)

function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    particles = MEState[]
    for i = 1:up.n
        s = rand(d)
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

function reweight(up::MEBeliefUpdater, b::MEBelief, particles::Vector{MEState},
                a::MEAction, o::MEObservation)
    po_s = Float64[]
    bore_coords = particles[1].bore_coords
    n = size(bore_coords)[2]
    bore_coords = hcat(bore_coords, [a.coords[1], a.coords[2]])
    ore_obs = [o.ore_quality for o in b.obs]
    push!(ore_obs, o.ore_quality)
    K = zeros(Float64, n+1, n+1)
    marginal_var = 0.006681951232101568
    # marginal_var = 1.0
    for i = 1:n+1
        for j = 1:n+1
            K[i, j] = clamp(marginal_var - variogram(bore_coords[:, i], bore_coords[:, j], up.m.variogram[6], marginal_var), 0.0, Inf)
            if i == j
                K[i, j] += 1e-3
            end
        end
    end
    for s in particles
        mainbody_cov = [s.var 0.0; 0.0 s.var]
        mainbody_dist = MvNormal(up.m.mainbody_loc, mainbody_cov)
        mainbody_max = 1.0/(2*π*s.var)

        w = 1.0
        o_n = zeros(Float64, n+1)
        for i = 1:n+1
            o_mainbody = pdf(mainbody_dist, bore_coords[:, i])
            o_mainbody /= mainbody_max
            o_mainbody *= up.m.mainbody_weight
            o_n[i] = (ore_obs[i] - o_mainbody)*0.3/up.m.gp_weight
        end
        mu = zeros(Float64, n+1) .+ 0.3
        gp_dist = MvNormal(mu, K)
        w = pdf(gp_dist, o_n)
        push!(po_s, w)
    end
    po_s ./= sum(po_s) + 1e-6
    return po_s
end

function resample(up::MEBeliefUpdater, b::MEBelief, wp::Vector{Float64}, a::MEAction, o::MEObservation)
    sampled_states = sample(up.rng, b.particles, StatsBase.Weights(wp), up.n, replace=true)
    sampled_vars = [s.var for s in sampled_states]
    mainbody_vars = Float64[]
    # mainbody_vars = [s.var for s in sampled_states]
    mainbody_maps = Matrix{Float64}[]
    mainbody_maxs = Float64[]
    for mainbody_var in sampled_vars
        if mainbody_var ∈ mainbody_vars
            mainbody_var += randn()
            mainbody_var = clamp(mainbody_var, 0.0, Inf)
        end
        push!(mainbody_vars, mainbody_var)
        mainbody_map = zeros(Float64, Int(up.m.grid_dim[1]), Int(up.m.grid_dim[2]))
        cov = [mainbody_var 0.0; 0.0 mainbody_var]
        mvnorm = MvNormal(up.m.mainbody_loc, cov)
        for i = 1:up.m.grid_dim[1]
            for j = 1:up.m.grid_dim[2]
                mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
            end
        end
        max_lode = maximum(mainbody_map)
        push!(mainbody_maxs, max_lode)
        mainbody_map ./= max_lode
        mainbody_map .*= up.m.mainbody_weight
        push!(mainbody_maps, mainbody_map)
    end
    # o_gp = (o.ore_quality - o_mainbody/mainbody_max*m.mainbody_weight)*(0.6/m.gp_weight)
    ore_quals = Float64[o.ore_quality for o in b.obs]
    push!(ore_quals, o.ore_quality)
    if b.bore_coords isa Nothing
        ore_coords = zeros(Int64, 2, 0)
    else
        ore_coords = b.bore_coords
    end
    ore_coords = hcat(ore_coords, [a.coords[1], a.coords[2]])
    # rock_obs = RockObservations(ore_quals, ore_coords)
    particles = MEState[]
    gslib_dist = GSLIBDistribution(up.m)
    gslib_dist.data.coordinates = ore_coords
    for (j, mainbody_map) in enumerate(mainbody_maps)
        n_ore_quals = Float64[]
        mainbody_max = mainbody_maxs[j]
        for (i, ore_qual) in enumerate(ore_quals)
            # n_ore_qual = ore_qual - mainbody_map[ore_coords[1, i], ore_coords[2, i]]
            prior_ore = mainbody_map[ore_coords[1, i], ore_coords[2, i]]
            # n_ore_qual = (ore_qual - prior_ore./mainbody_max.*up.m.mainbody_weight).*(0.6/up.m.gp_weight) - 0.6
            n_ore_qual = (ore_qual - prior_ore).*(0.3/up.m.gp_weight)
            push!(n_ore_quals, n_ore_qual)
        end
        gslib_dist.data.ore_quals = n_ore_quals
        gp_ore_map = Base.rand(up.rng, gslib_dist)
        mean_gp = mean(gp_ore_map)
        gp_ore_map ./= 0.3
        gp_ore_map .*= up.m.gp_weight
        clamp!(gp_ore_map, 0.0, up.m.massive_threshold)
        mainbody_map_3d = repeat(mainbody_map, outer=(1, 1, 8))
        ore_map = gp_ore_map .+ mainbody_map_3d

        clamp!(ore_map, 0.0, 1.0)

        new_state = MEState(ore_map, mainbody_vars[j],
                gslib_dist.data.coordinates, false, false)
        push!(particles, new_state)
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)
    wp = reweight(up, b, b.particles, a, o)
    pp = resample(up, b, wp, a, o)
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
