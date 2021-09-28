module MineralExploration

using POMDPs
using POMDPModelTools
using Random
using Plots
using Parameters
using StatsBase

export
        CCSState,
        CCSObservation,
        RockObservations
include("common.jl")

export
        GSLIBDistribution
include("gslib.jl")


export
        RockObservations,
        CCSState,
        CCSObservation,
        CCSPOMDP,
        POMDPSpecification,
        initialize_data!
include("pomdp.jl")

export
        ReservoirBelief,
        ReservoirBeliefUpdater,
        solve_gp
include("beliefs.jl")

export
        TestPOMDP2D
include("2d_pomdp_test.jl")

end
