
struct TestPOMDP2D <: CCSPOMDP
    spec::POMDPSpecification
    max_n_wells::Int64 # From spec, for convenience
    max_placement_t::Float64 # From spec, for convenience
    initial_data::RockObservations # Initial rock observations
    delta::Int64 # Minimum distance between wells
    injection_rate::Float64 # Fixed well injection rate
    rng::AbstractRNG
end

function TestPOMDP2D(s::POMDPSpecification; rng::AbstractRNG=Random.GLOBAL_RNG)
    TestPOMDP2D(s, s.max_n_wells, s.max_placement_t, s.initial_data, s.delta,
        s.injection_rate, rng)
end

function POMDPs.gen(m::TestPOMDP2D, s::CCSState, a::CartesianIndex, rng::AbstractRNG)
    poro_obs = s.porosity[a[1], a[2], 1]
    obs = CCSObservation(Float64[], Float64[], poro_obs)
    r = s.perm_z[a]
    a = reshape(Int64[a[1] a[2]], 2, 1)
    coords_p = [s.well_coords a]
    sp = CCSState(s.porosity, s.perm_xy, s.perm_z, coords_p, s.obs_well_coord)
    return (sp=sp, o=obs, r=r)
end

function POMDPs.actions(m::TestPOMDP2D)
    idxs = CartesianIndices(m.spec.grid_dim)
    reshape(collect(idxs), prod(m.spec.grid_dim))
end

function POMDPs.actions(m::TestPOMDP2D, s::CCSState)
    action_set = Set(POMDPs.actions(m))
    for i=1:size(s.well_coords)[2]
        coord = s.well_coords[:, i]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

function POMDPs.actions(m::TestPOMDP2D, b::ReservoirBelief)
    action_set = Set(POMDPs.actions(m))
    n_initial = length(m.initial_data)
    n_obs = size(b.rock_belief.data.coordinates)[2] - n_initial
    for i=1:n_obs
        coord = b.rock_belief.data.coordinates[:, i + n_initial]
        x = Int64(coord[1])
        y = Int64(coord[2])
        keepout = Set(collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta))))
        setdiff!(action_set, keepout)
    end
    collect(action_set)
end

POMDPs.discount(::TestPOMDP2D) = 0.99

function POMDPs.isterminal(m::TestPOMDP2D, s::CCSState)
    n_initial = length(m.initial_data)
    n_wells = size(s.well_coords)[2] - n_initial
    return n_wells >= m.max_n_wells
end

function POMDPModelTools.obs_weight(p::TestPOMDP2D, s::CCSState,
                            a::CartesianIndex, sp::CCSState, o::CCSObservation)
    o_true = sp.porosity[a]
    if o_true == o
        return 1.0
    else
        return 0.0
    end
end
