struct MEBeliefUpdater{G} <: POMDPs.Updater
    m::MineralExplorationPOMDP
    geostats::G
    n::Int64
    noise::Float64
    abc::Bool
    abc_ϵ::Float64
    rng::AbstractRNG
end

function MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int64, noise::Float64=1.0; abc::Bool=false, abc_ϵ::Float64=1e-4)
    geostats = m.geodist_type(m)
    return MEBeliefUpdater(m, geostats, n, noise, abc, abc_ϵ, m.rng)
end


struct MEBelief{G}
    particles::Vector{MEState} # Vector of vars & lode maps
    rock_obs::RockObservations
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
    stopped::Bool
    decided::Bool
    geostats::G #GSLIB or GeoStats
    up::MEBeliefUpdater ## include the belief updater
end

# Ensure MEBeliefs can be compared when adding them to dictionaries (using `hash`, `isequal` and `==`)
Base.hash(r::RockObservations, h::UInt) = hash(Tuple(getproperty(r, p) for p in propertynames(r)), h)
Base.isequal(r1::RockObservations, r2::RockObservations) = all(isequal(getproperty(r1, p), getproperty(r2, p)) for p in propertynames(r1))
Base.:(==)(r1::RockObservations, r2::RockObservations) = isequal(r1, r2)

Base.hash(g::GeoStatsDistribution, h::UInt) = hash(Tuple(getproperty(g, p) for p in propertynames(g)), h)
Base.isequal(g1::GeoStatsDistribution, g2::GeoStatsDistribution) = all(isequal(getproperty(g1, p), getproperty(g2, p)) for p in propertynames(g1))
Base.:(==)(g1::GeoStatsDistribution, g2::GeoStatsDistribution) = isequal(g1, g2)

Base.hash(b::MEBelief, h::UInt) = hash(Tuple(getproperty(b, p) for p in propertynames(b)), h)
Base.isequal(b1::MEBelief, b2::MEBelief) = all(isequal(getproperty(b1, p), getproperty(b2, p)) for p in propertynames(b1))
Base.:(==)(b1::MEBelief, b2::MEBelief) = isequal(b1, b2)


function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    particles = rand(up.rng, d, up.n)
    init_rocks = up.m.initial_data
    rock_obs = RockObservations(init_rocks.ore_quals, init_rocks.coordinates)
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(particles, rock_obs, acts, obs, false, false, up.geostats, up)
end

# TODO: ParticleFilters.particles
particles(b::MEBelief) = b.particles
# TODO: ParticleFilters.support
POMDPs.support(b::MEBelief) = POMDPs.support(particles(b))

function calc_K(geostats::GeoDist, rock_obs::RockObservations)
    if isa(geostats, GeoStatsDistribution)
        pdomain = geostats.domain
        γ = geostats.variogram
    else
        pdomain = CartesianGrid{Int64}(geostats.grid_dims[1], geostats.grid_dims[2])
        γ = SphericalVariogram(sill=geostats.sill, range=geostats.variogram[6], nugget=geostats.nugget)
    end
    # table = DataFrame(ore=rock_obs.ore_quals .- geostats.mean)
    domain = PointSet(rock_obs.coordinates)
    # pdata = georef(table, domain)
    # vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    # dlocs = Int64[m[1] for m in vmapping]
    # 𝒟d = [centroid(pdomain, i) for i in dlocs]
    𝒟d = [GeoStats.Point(p.coords[1] + 0.5, p.coords[2] + 0.5) for p in domain]
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
        mb_map = s.mainbody_map
        o_n = zeros(Float64, n)
        for i = 1:n
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]]
            o_n[i] = ore_obs[i] - o_mainbody
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws .+= 1e-6
    ws ./= sum(ws)
    return ws
end

function resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64},
                geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation;
                apply_perturbation=true, n=up.n)
    sampled_particles = sample(up.rng, particles, StatsBase.Weights(wp), n, replace=true)
    mainbody_params = []
    mainbody_maps = Array{Float64, 3}[]
    particles = MEState[]
    x = nothing
    ore_quals = deepcopy(rock_obs.ore_quals)
    for s in sampled_particles
        mainbody_param = s.mainbody_params
        mainbody_map = s.mainbody_map
        ore_map = s.ore_map
        if apply_perturbation
            if mainbody_param ∈ mainbody_params
                mainbody_map, mainbody_param = perturb_sample(up.m.mainbody_gen, mainbody_param, up.noise)
                max_lode = maximum(mainbody_map)
                mainbody_map ./= max_lode
                mainbody_map .*= up.m.mainbody_weight
                mainbody_map = reshape(mainbody_map, up.m.grid_dim)
                # clamp!(ore_map, 0.0, 1.0)
            end
        end
        n_ore_quals = Float64[]
        for (i, ore_qual) in enumerate(ore_quals)
            prior_ore = mainbody_map[rock_obs.coordinates[1, i], rock_obs.coordinates[2, i], 1]
            n_ore_qual = (ore_qual - prior_ore)
            push!(n_ore_quals, n_ore_qual)
        end
        geostats.data.ore_quals = n_ore_quals
        if apply_perturbation
            gp_ore_map = Base.rand(up.rng, geostats)
            ore_map = gp_ore_map .+ mainbody_map
        end
        rock_obs_p = RockObservations(rock_obs.ore_quals, rock_obs.coordinates)
        sp = MEState(ore_map, mainbody_param, mainbody_map, rock_obs_p,
                    o.stopped, o.decided)
        push!(mainbody_params, mainbody_param)
        push!(particles, sp)
    end
    return particles
end

function update_particles(up::MEBeliefUpdater, particles::Vector{MEState},
                          geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    wp = reweight(up, geostats, particles, rock_obs)
    pp = resample(up, particles, wp, geostats, rock_obs, a, o)
    return pp
end


function update_particles_perturbed_inject(up::MEBeliefUpdater, particles::Vector{MEState},
                                           geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    m = 50 # TODO: parameterize `m`
    wp = reweight(up, geostats, particles, rock_obs)
    injected_particles = resample(up, particles, wp, geostats, rock_obs, a, o; apply_perturbation=true, n=m)
    particles = vcat(particles, injected_particles)
    wp2 = reweight(up, geostats, particles, rock_obs)
    pp = resample(up, particles, wp2, geostats, rock_obs, a, o; apply_perturbation=false)
    return pp
end


function reweight_abc(up::MEBeliefUpdater, particles::Vector, rock_obs::RockObservations)
    ws = Float64[]
    ϵ = up.abc_ϵ
    dist_mse(x, y) = (x - y)^2
    dist_abs(x, y) = abs(x - y)
    rho = dist_mse
    ore_quals = deepcopy(rock_obs.ore_quals)
    actions = deepcopy(rock_obs.coordinates)
    for particle in particles
        w = 1
        for a in eachcol(actions)
            for o in ore_quals
                b_o = particle.ore_map[a[1], a[2]]
                w = rho(b_o, o)
                w *= w ≤ ϵ ? w : 0 # 1e-8 # acceptance tolerance
            end
        end
        push!(ws, w)
    end
    ws .+= 1e-6
    normalize!(ws, 1)
    return ws
end

function update_particles_abc(up::MEBeliefUpdater, particles::Vector{MEState},
                              geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    wp = reweight_abc(up, particles, rock_obs)
    pp = resample(up, particles, wp, geostats, rock_obs, a, o; apply_perturbation=false)
    return pp
end

function inject_particles(up::MEBeliefUpdater, n::Int64)
    d = POMDPs.initialstate_distribution(up.m) # TODO. Keep as part of `MEBeliefUpdater`
    return rand(up.rng, d, n)
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
                       a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_particles = MEState[] # MEState[p for p in b.particles]
        for p in b.particles
            s = MEState(p.ore_map, p.mainbody_params, p.mainbody_map, p.rock_obs, o.stopped, o.decided)
            push!(bp_particles, s)
        end
        bp_rock = RockObservations(ore_quals=deepcopy(b.rock_obs.ore_quals),
                                coordinates=deepcopy(b.rock_obs.coordinates))
        # TODO Make this a little more general in future
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, bp_rock,
                                            b.geostats.domain, b.geostats.mean,
                                            b.geostats.variogram, b.geostats.lu_params)
        elseif up.m.geodist_type == GSLIBDistribution
            bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                                            bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                                            b.geostats.variogram, b.geostats.target_histogram_file,
                                            b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                                            b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                                            b.geostats.transform_data, b.geostats.mn,
                                            b.geostats.sz)
        end
    else
        bp_rock = deepcopy(b.rock_obs)
        bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]])
        push!(bp_rock.ore_quals, o.ore_quality)
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock),
                                            b.geostats.domain, b.geostats.mean,
                                            b.geostats.variogram, b.geostats.lu_params)
            update!(bp_geostats, bp_rock)
        elseif up.m.geodist_type == GSLIBDistribution
            bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                                            bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                                            b.geostats.variogram, b.geostats.target_histogram_file,
                                            b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                                            b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                                            b.geostats.transform_data, b.geostats.mn,
                                            b.geostats.sz)
        end
        f_update_particles = up.abc ? update_particles_abc : update_particles
        bp_particles = f_update_particles(up, b.particles, bp_geostats, bp_rock, a, o)
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
                    bp_decided, bp_geostats, up)
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
        action_set = OrderedSet(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(b.rock_obs.ore_quals)
            n_obs = length(b.rock_obs.ore_quals) - n_initial
            if m.max_movement != 0 && n_obs > 0
                d = m.max_movement
                drill_s = b.rock_obs.coordinates[:,end]
                x = drill_s[1]
                y = drill_s[2]
                reachable_coords = CartesianIndices((x-d:x+d,y-d:y+d))
                reachable_acts = MEAction[]
                for coord in reachable_coords
                    dx = abs(x - coord[1])
                    dy = abs(y - coord[2])
                    d2 = sqrt(dx^2 + dy^2)
                    if d2 <= d
                        push!(reachable_acts, MEAction(coords=coord))
                    end
                end
                push!(reachable_acts, MEAction(type=:stop))
                reachable_acts = OrderedSet(reachable_acts)
                # reachable_acts = OrderedSet([MEAction(coords=coord) for coord in collect(reachable_coords)])
                action_set = intersect(reachable_acts, action_set)
            end
            for i=1:n_obs
                coord = b.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = OrderedSet([MEAction(coords=coord) for coord in keepout])
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
    s = rand(m.rng, b.sr_belief.dist)[1]
    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = OrderedSet(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(s.rock_obs.ore_quals)
            n_obs = length(s.rock_obs.ore_quals) - n_initial
            if m.max_movement != 0 && n_obs > 0
                d = m.max_movement
                drill_s = s.rock_obs.coordinates[:,end]
                x = drill_s[1]
                y = drill_s[2]
                reachable_coords = CartesianIndices((x-d:x+d,y-d:y+d))
                reachable_acts = MEAction[]
                for coord in reachable_coords
                    dx = abs(x - coord[1])
                    dy = abs(y - coord[2])
                    d2 = sqrt(dx^2 + dy^2)
                    if d2 <= d
                        push!(reachable_acts, MEAction(coords=coord))
                    end
                end
                push!(reachable_acts, MEAction(type=:stop))
                reachable_acts = OrderedSet(reachable_acts)
                # reachable_acts = OrderedSet([MEAction(coords=coord) for coord in collect(reachable_coords)])
                action_set = intersect(reachable_acts, action_set)
            end
            for i=1:n_obs
                coord = s.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = OrderedSet([MEAction(coords=coord) for coord in keepout])
                setdiff!(action_set, keepout_acts)
            end
            if n_obs < m.min_bores
                delete!(action_set, MEAction(type=:stop))
            end
        elseif m.min_bores > 0
            delete!(action_set, MEAction(type=:stop))
        end
        # delete!(action_set, MEAction(type=:mine))
        # delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)
    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = OrderedSet(POMDPs.actions(m))
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

function Plots.plot(b::MEBelief, t=nothing; cmap=:viridis)
    mean, var = summarize(b)
    if t == nothing
        mean_title = "belief mean"
        std_title = "belief std"
    else
        mean_title = "belief mean t=$t"
        std_title = "belief std t=$t"
    end
    xl = (1,size(mean,1))
    yl = (1,size(mean,2))
    fig1 = heatmap(mean[:,:,1], title=mean_title, fill=true, clims=(0.0, 1.0), legend=:none, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig2 = heatmap(sqrt.(var[:,:,1]), title=std_title, fill=true, legend=:none, clims=(0.0, 0.2), ratio=1, c=cmap, xlims=xl, ylims=yl)
    if !isempty(b.rock_obs.ore_quals)
        x = b.rock_obs.coordinates[2, :]
        y = b.rock_obs.coordinates[1, :]
        plot!(fig1, x, y, seriestype = :scatter)
        plot!(fig2, x, y, seriestype = :scatter)
        n = length(b.rock_obs)
        if n > 1
            for i = 1:n-1
                x = b.rock_obs.coordinates[2, i:i+1]
                y = b.rock_obs.coordinates[1, i:i+1]
                plot!(fig1, x, y, arrow=:closed, color=:black)
            end
        end
    end
    fig = plot(fig1, fig2, layout=(1,2), size=(600,250))
    return fig
end


data_skewness(D) = [skewness(D[x,y,1:end-1]) for x in 1:size(D,1), y in 1:size(D,2)]
data_kurtosis(D) = [kurtosis(D[x,y,1:end-1]) for x in 1:size(D,1), y in 1:size(D,2)]


function convert2data(b::MEBelief)
    states = cat([p.ore_map[:,:,1] for p in particles(b)]..., dims=3)
    observations = zeros(size(states)[1:2])
    for (i,a) in enumerate(b.acts)
        if a.type == :drill
            x, y = a.coords.I
            observations[x,y] = b.obs[i].ore_quality
        end
    end
    return cat(states, observations; dims=3)
end


function get_input_representation(b::MEBelief)
    D = convert2data(b)
    μ = mean(D[:,:,1:end-1], dims=3)[:,:,1]
    σ² = std(D[:,:,1:end-1], dims=3)[:,:,1]
    sk = data_skewness(D)
    kurt = data_kurtosis(D)
    obs = D[:,:,end]
    return cat(μ, σ², sk, kurt, obs; dims=3)
end


plot_input_representation(b::MEBelief) = plot_input_representation(get_input_representation(b))
function plot_input_representation(B::Array{<:Real, 3})
    μ = B[:,:,1]
    σ² = B[:,:,2]
    sk = B[:,:,3]
    kurt = B[:,:,4]
    obs = B[:,:,5]
    xl = (1,size(μ,1))
    yl = (1,size(μ,2))
    cmap = :viridis
    fig1 = heatmap(μ, title="mean", fill=true, clims=(0, 1), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig2 = heatmap(σ², title="stdev", fill=true, clims=(0.0, 0.2), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig3 = heatmap(sk, title="skewness", fill=true, clims=(-1, 1), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig4 = heatmap(kurt, title="kurtosis", fill=true, clims=(-3, 3), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig5 = heatmap(obs, title="obs", fill=true, clims=(0, 1), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    return plot(fig1, fig2, fig3, fig4, fig5, layout=(1,5), size=(300*5,250), margin=3Plots.mm)
end
