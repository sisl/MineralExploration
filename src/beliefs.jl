# struct ReservoirBelief #TODO Add well locations
#     rock_belief::GSLIBDistribution
#     pressure_history::Vector{Float64}
#     saturation_history::Vector{Float64}
#     well_coords::Union{Nothing, Matrix{Int64}}
#     obs_well_coords::Union{Nothing, Vector{Int64}}
# end
#
# struct ReservoirBeliefUpdater <: POMDPs.Updater
#     spec::POMDPSpecification
# end
#
# function POMDPs.update(::ReservoirBeliefUpdater, b::ReservoirBelief,
#                     action::CartesianIndex, obs::CCSObservation)1
# end
#
# function POMDPs.initialize_belief(up::ReservoirBeliefUpdater, s::CCSState)
#     rock_belief = GSLIBDistribution(up.spec)
#     init_well_coords = rock_belief.data.coordinates
#     ReservoirBelief(rock_belief, Float64[], Float64[], init_well_coords, nothing)
# end
#
# function POMDPs.initialize_belief(up::ReservoirBeliefUpdater, b::ReservoirBelief)
#     ReservoirBelief(b.rock_belief, b.pressure_history, b.saturation_history, b.well_coords, b.obs_well_coords)
# end
#
# function Base.rand(rng::AbstractRNG, b::ReservoirBelief)
#     rock_state = rand(rng, b.rock_belief)
#     CCSState(rock_state[1], rock_state[2], rock_state[3], b.well_coords, b.obs_well_coords)
# end
#
# Base.rand(b::ReservoirBelief) = Base.rand(Random.GLOBAL_RNG, b)
#
# function Plots.plot(b::ReservoirBelief) # TODO add well plots
#     poro_mean, poro_var = solve_gp(b.rock_belief)
#     # println(size(poro_mean))
#     # STOP
#     fig1 = heatmap(poro_mean[:,:,1], title="Porosity Mean", fill=true, clims=(0.0, 0.38))
#     fig2 = heatmap(poro_var[:,:,1], title="Porosity Variance")
#     fig = plot(fig1, fig2, layout=(1,2))
#     return fig
# end

function POMDPs.obs_weight(m::MineralExplorationPOMDP, s::MEState,
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
