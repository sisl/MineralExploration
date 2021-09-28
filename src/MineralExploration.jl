module MineralExploration

using POMDPs
using POMDPModelTools
using Random
using Plots
using CSV
using Parameters
using StatsBase
using Distributions

export
        MEState,
        MEObservation,
        RockObservations
include("common.jl")

export
        GSLIBDistribution
include("gslib.jl")


export
        MineralExplorationPOMDP,
        MEInitStateDist,
        initialize_data!
include("pomdp.jl")

# export
#         ReservoirBelief,
#         ReservoirBeliefUpdater,
#         solve_gp
# include("beliefs.jl")

end
