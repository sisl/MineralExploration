
struct MEBelief
    bore_coords::Union{Nothing, Matrix{Int64}}
    stopped::Bool
    particles::WeightedParticleBelief
end

struct NullResampler end

struct MEBeliefUpdater <: POMDPs.Updater
    m::MineralExplorationPOMDP
    n::Int64
    pf::BasicParticleFilter
    function MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
        pf = BasicParticleFilter(m, NullResampler(), n, rng)
        new(m, n, pf)
    end
end

function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    S = POMDPs.statetype(up.m)
    particles = S[]
    weights = Float64[]
    w0 = 1.0/up.n
    for i = 1:up.n
        s = rand(d)
        push!(particles, s)
        push!(weights, w0)
    end
    particle_set = WeightedParticleBelief(particles, weights)
    return MEBelief(nothing, false, particle_set)
end

function ParticleFilters.predict!(pm, m::MineralExplorationPOMDP, b::WeightedParticleBelief, ::Any, ::AbstractRNG)
    for p in b.particles
        push!(pm, p)
    end
end

function ParticleFilters.reweight!(wm, m::MineralExplorationPOMDP, b::WeightedParticleBelief,
                            a::MEAction, pm, y::MEObservation, rng::AbstractRNG=Random.GLOBAL_RNG)
    b0 = Float64[w for w in b.weights]
    po_s = Float64[]
    for s in pm
        push!(po_s, obs_weight(m, s, a, s, y))
    end
    bp = b0.*po_s
    bp ./= sum(bp)
    for w in bp
        push!(wm, w)
    end
end

function ParticleFilters.resample(::NullResampler, bp::WeightedParticleBelief, ::AbstractRNG)
    return WeightedParticleBelief(bp.particles, bp.weights)
end

function ParticleFilters.update(pf::BasicParticleFilter, b::WeightedParticleBelief,
                                a::Any, o::Any)
    S = typeof(b.particles[1])
    pm = S[]
    wm = Float64[]
    ParticleFilters.predict!(pm, pf.predict_model, b, a, pf.rng)
    ParticleFilters.reweight!(wm, pf.reweight_model, b, a, pm, o, pf.rng)
    bp = WeightedParticleBelief(pm, wm)
    bp = ParticleFilters.resample(pf.resampler, bp, pf.rng)
    return bp
end

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
    bp_particles = ParticleFilters.update(up.pf, b.particles, a, o)
    return MEBelief(bp_coords, bp_stopped, bp_particles)
end

function obs_weight(m::MineralExplorationPOMDP, s::MEState,
                    a::MEAction, sp::MEState, o::MEObservation)
    w = 0.0
    if a.type != :drill
        w = o.ore_quality == nothing ? 1.0 : 0.0
    else
        ore = s.ore_map[:,:,1][a.coords]
        dist = Normal(ore, m.obs_noise_std)
        w = pdf(dist, o.ore_quality)
    end
    return w
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
