struct ReservoirBelief #TODO Add well locations
    rock_belief::GSLIBDistribution
    pressure_history::Vector{Float64}
    saturation_history::Vector{Float64}
    well_coords::Union{Nothing, Matrix{Int64}}
    obs_well_coords::Union{Nothing, Vector{Int64}}
end

struct ReservoirBeliefUpdater <: POMDPs.Updater
    spec::POMDPSpecification
end

function POMDPs.update(::ReservoirBeliefUpdater, b::ReservoirBelief,
                    action::CartesianIndex, obs::CCSObservation)
    act_array = reshape(Float64[action[1] action[2]], 2, 1)
    if b.obs_well_coords == nothing
        obs_well_coords = [action[1], action[2]]
        well_coords = b.well_coords
    else
        obs_well_coords = b.obs_well_coords
        well_coords = hcat(b.well_coords, [action[1], action[2]])
    end
    bp = ReservoirBelief(b.rock_belief, b.pressure_history, b.saturation_history, well_coords, obs_well_coords)
    bp.rock_belief.data.coordinates = [b.rock_belief.data.coordinates act_array] # This is specific to our problem formulation where action:=location
    bp.rock_belief.data.porosity = [bp.rock_belief.data.porosity; obs.porosity]
    append!(bp.pressure_history, obs.pressure_history)
    append!(bp.saturation_history, obs.saturation_history)
    return bp
end

function POMDPs.initialize_belief(up::ReservoirBeliefUpdater, s::CCSState)
    rock_belief = GSLIBDistribution(up.spec)
    init_well_coords = rock_belief.data.coordinates
    ReservoirBelief(rock_belief, Float64[], Float64[], init_well_coords, nothing)
end

function POMDPs.initialize_belief(up::ReservoirBeliefUpdater, b::ReservoirBelief)
    ReservoirBelief(b.rock_belief, b.pressure_history, b.saturation_history, b.well_coords, b.obs_well_coords)
end

function Base.rand(rng::AbstractRNG, b::ReservoirBelief)
    rock_state = rand(rng, b.rock_belief)
    CCSState(rock_state[1], rock_state[2], rock_state[3], b.well_coords, b.obs_well_coords)
end

Base.rand(b::ReservoirBelief) = Base.rand(Random.GLOBAL_RNG, b)

function calculate_obs_weight(o::Matrix{Float64}, o_true::Matrix{Float64};
                            l::Float64=1.0)
    d = mean((o - o_true).^2)
    return exp(-d/(2*l^2))
end

function Plots.plot(b::ReservoirBelief) # TODO add well plots
    poro_mean, poro_var = solve_gp(b.rock_belief)
    # println(size(poro_mean))
    # STOP
    fig1 = heatmap(poro_mean[:,:,1], title="Porosity Mean", fill=true, clims=(0.0, 0.38))
    fig2 = heatmap(poro_var[:,:,1], title="Porosity Variance")
    fig = plot(fig1, fig2, layout=(1,2))
    return fig
end
