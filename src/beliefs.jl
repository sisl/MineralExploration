
struct MEBelief
    bore_coords::Union{Nothing, Matrix{Int64}}
    stopped::Bool
    weights::Vector{Float64}
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
    weights = Float64[]
    w0 = 1.0/up.n
    for i = 1:up.n
        s = rand(d)
        push!(particles, s)
        push!(weights, w0)
    end
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(nothing, false, weights, particles, acts, obs)
end

# function ParticleFilters.reweight!(wm, m::MineralExplorationPOMDP, b::WeightedParticleBelief,
#                             a::MEAction, pm, y::MEObservation, rng::AbstractRNG=Random.GLOBAL_RNG)
#     b0 = Float64[w for w in b.weights]
#     po_s = Float64[]
#     for s in pm
#         push!(po_s, obs_weight(m, s, a, s, y))
#     end
#     bp = b0.*po_s
#     bp ./= sum(bp)
#     for w in bp
#         push!(wm, w)
#     end
# end

function obs_weight(m::MineralExplorationPOMDP, s::MEState,
                    a::MEAction, sp::MEState, o::MEObservation)
    w = 0.0
    if a.type != :drill
        w = o.ore_quality == nothing ? 1.0 : 0.0
    else
        mainbody_cov = [s.var 0.0; 0.0 s.var]
        mainbody_dist = MvNormal(m.mainbody_loc, mainbody_cov)
        o_mainbody = pdf(mainbody_dist, [float(a.coords[1]), float(a.coords[2])]) # TODO

        mainbody_max = 1.0/(2*π*s.var)
        o_gp = (o.ore_quality - o_mainbody/mainbody_max*m.mainbody_weight)*(0.6/m.gp_weight)
        # o_gp = o.ore_quality - o_mainbody
        # println(s.bore_coords)
        if s.bore_coords isa Nothing || size(s.bore_coords)[2] == 0
            mu = 0.6
            sigma = 1.0
        else
            gslib_dist = GSLIBDistribution(m)
            prior_ore = Float64[]
            for i =1:size(s.bore_coords[2])
                push!(prior_ore, ore_map[s.bore_coords[1, i], s.bore_coords[2, i], 1])
            end
            prior_obs = RockObservations(prior_ore, s.bore_coords)
            μ, σ² = kriging(gslib_dist, prior_obs)
            mu = μ[a.coords]
            sigma = sqrt(σ²[a.coords])
        end
        point_dist = Normal(mu, sigma)
        w = pdf(point_dist, o_gp)
    end
    return w
end

function reweight(up::MEBeliefUpdater, particles::Vector{MEState},
                weights::Vector{Float64}, a::MEAction, o::MEObservation)
    b0 = Float64[w for w in weights]
    po_s = Float64[]
    for s in particles
        push!(po_s, obs_weight(up.m, s, a, s, o))
    end
    bp = b0.*po_s
    bp ./= sum(bp)
    return bp
end

function resample(up::MEBeliefUpdater, b::MEBelief, wp::Vector{Float64}, a::MEAction, o::MEObservation)
    sampled_states = sample(up.rng, b.particles, StatsBase.Weights(wp), up.n, replace=true)
    mainbody_vars = [s.var for s in sampled_states]
    mainbody_maps = Matrix{Float64}[]
    for mainbody_var in mainbody_vars
        mainbody_map = zeros(Float64, Int(up.m.grid_dim[1]), Int(up.m.grid_dim[2]))
        cov = [mainbody_var 0.0; 0.0 mainbody_var]
        mvnorm = MvNormal(up.m.mainbody_loc, cov)
        for i = 1:up.m.grid_dim[1]
            for j = 1:up.m.grid_dim[2]
                mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
            end
        end
        max_lode = maximum(mainbody_map)
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
        for (i, ore_qual) in enumerate(ore_quals)
            n_ore_qual = ore_qual - mainbody_map[ore_coords[1, i], ore_coords[2, i]]
            push!(n_ore_quals, n_ore_qual)
        end
        gslib_dist.data.ore_quals = n_ore_quals
        gp_ore_map = Base.rand(up.rng, gslib_dist)
        mean_gp = mean(gp_ore_map)
        gp_ore_map ./= mean_gp
        gp_ore_map .*= up.m.gp_weight
        clamp!(gp_ore_map, 0.0, up.m.massive_threshold)
        mainbody_map = repeat(mainbody_map, outer=(1, 1, 8))
        # println(size(gp_ore_map))
        # println(size(mainbody_map))
        ore_map = gp_ore_map + mainbody_map
        clamp!(ore_map, 0.0, 1.0)
        new_state = MEState(ore_map, mainbody_vars[j],
                gslib_dist.data.coordinates, false, false)
        push!(particles, new_state)
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)
    wp = reweight(up, b.particles, b.weights, a, o)
    pp = resample(up, b, wp, a, o)
    return pp
end
# function ParticleFilters.update(pf::BasicParticleFilter, b::WeightedParticleBelief,
#                                 a::Any, o::Any)
#
#     # S = typeof(b.particles[1])
#     # pm = S[]
#     # wm = Float64[]
#     # ParticleFilters.predict!(pm, pf.predict_model, b, a, pf.rng)
#     # ParticleFilters.reweight!(wm, pf.reweight_model, b, a, pm, o, pf.rng)
#     # bp = WeightedParticleBelief(pm, wm)
#     # bp = ParticleFilters.resample(pf.resampler, bp, pf.rng)
#     # return bp
# end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
                            a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_coords = b.bore_coords
        bp_stopped = true
    else
        if b.bore_coords == nothing
            bp_coords = reshape([a.coords[1], a.coords[2]], 2, 1)
        else
            bp_coords = hcat(b.bore_coords, [a.coords[1], a.coords[2]])
        end
        bp_stopped = o.stopped
    end
    bp_particles = update_particles(up, b, a, o)
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
    return MEBelief(bp_coords, bp_stopped, bp_weights, bp_particles, bp_acts, bp_obs)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    s0 = rand(rng, b.particles)
    return MEState(s0.ore_map, s0.bore_coords, b.stopped, false)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    (x, y, z) = size(b.particles.particles[1].ore_map)
    μ = zeros(Float64, x, y, z)
    for (i, p) in enumerate(b.particles.particles)
        w = b.particles.weights[i]
        ore_map = convert(Array{Float64, 3}, p.ore_map)
        μ .+= ore_map .* w
    end
    σ² = zeros(Float64, x, y, z)
    for (i, p) in enumerate(b.particles.particles)
        w = b.particles.weights[i]
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
        # n_initial = length(m.initial_data)
        # if s.bore_coords != nothing
        #     n_obs = size(s.bore_coords)[2] - n_initial
        #     for i=1:n_obs
        #         coord = s.bore_coords[:, i + n_initial]
        #         x = Int64(coord[1])
        #         y = Int64(coord[2])
        #         keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
        #         keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
        #         setdiff!(action_set, keepout_acts)
        #     end
        # end
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
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
