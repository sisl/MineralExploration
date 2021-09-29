
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
                            a::Union{Symbol, CartesianIndex}, pm, y::MEObservation, rng::AbstractRNG=Random.GLOBAL_RNG)
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
                            a::Union{Symbol, CartesianIndex}, o::MEObservation)
    if a == :stop
        bp_coords = b.bore_coodrs
        bp_stopped = true
    else
        if b.bore_coords == nothing
            bp_coords = reshape([a[1], a[2]], 2, 1)
        else
            bp_coords = hcat(b.bore_coords, [a[1], a[2]])
        end
        bp_stopped = b.stopped
    end
    bp_particles = ParticleFilters.update(up.pf, b.particles, a, o)
    return MEBelief(bp_coords, bp_stopped, bp_particles)
end

function obs_weight(m::MineralExplorationPOMDP, s::MEState,
                    a::Union{Symbol, CartesianIndex}, sp::MEState, o::MEObservation)
    w = 0.0
    if a == :stop
        w = o.ore_quality == nothing ? 1.0 : 0.0
    else
        ore = s.ore_map[a]
        dist = Normal(ore, m.obs_noise_std)
        w = pdf(dist, o.ore_quality)
    end
    return w
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    rand(rng, b.particles)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    μ = b.particles.particles[1].ore_map.*0.0
    for (i, p) in enumerate(b.particles.particles)
        w = b.particles.weights[i]
        μ .+= p.ore_map .* w
    end
    σ² = b.particles.particles[1].ore_map.*0.0
    for (i, p) in enumerate(b.particles.particles)
        w = b.particles.weights[i]
        σ² += w*(p.ore_map - μ).^2
    end
    return (μ, σ²)
end

function Plots.plot(b::MEBelief) # TODO add well plots
    mean, var = summarize(b)
    fig1 = heatmap(mean[:,:,1], title="Belief Mean", fill=true, clims=(0.0, 1.0))
    fig2 = heatmap(sqrt.(var[:,:,1]), title="Belief StdDev", fill=true, clims=(0.0, 0.25))
    fig = plot(fig1, fig2, layout=(1,2))
    return fig
end
