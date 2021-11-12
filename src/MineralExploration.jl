module MineralExploration

using POMDPs
using POMDPModelTools
using POMCPOW
using BeliefUpdaters
using Random
using Plots
using GeoStats
using DataFrames
using Parameters
using StatsBase
using Statistics
using Distributions
using LinearAlgebra

export
        MEState,
        MEObservation,
        MEAction,
        RockObservations
include("common.jl")

export
        GeoStatsDistribution,
        kriging
include("geostats.jl")


export
        MineralExplorationPOMDP,
        MEInitStateDist,
        initialize_data!
include("pomdp.jl")

export
        MEBelief,
        MEBeliefUpdater
include("beliefs.jl")

export
        NextActionSampler,
        ExpertPolicy,
        RandomSolver,
        GridPolicy,
        leaf_estimation
include("solver.jl")

export
        GPNextAction
include("action_selection.jl")

end
