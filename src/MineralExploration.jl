module MineralExploration

using BeliefUpdaters
using CSV
using DataFrames
using DelimitedFiles
using Distributions
using Distances # for KL and JS
using GeoStats
using ImageFiltering
using Infiltrator # for debugging
using Interpolations
using JLD
using JSON
using KernelDensity
using Luxor
using LinearAlgebra
using MCTS
using Parameters
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using POMCPOW
using POMDPModelTools
using POMDPSimulators
using POMDPs
using Random
using StatsBase
using StatsPlots
using Statistics
using OrderedCollections


export
        MEState,
        MEObservation,
        MEAction,
        RockObservations,
        GeoDist,
        MineralExplorationPOMDP,
        MEInitStateDist,
        MEBelief,
        MainbodyGen
include("common.jl")

export
        GeoStatsDistribution,
        kriging
include("geostats.jl")

export
        GSLIBDistribution,
        kriging
include("gslib.jl")

export
        SingleFixedNode,
        SingleVarNode,
        MultiVarNode
include("mainbody.jl")

export
        MEBeliefUpdater,
        particles,
        support,
        get_input_representation,
        plot_input_representation
include("beliefs.jl")

export
        initialize_data!,
        high_fidelity_obs
include("pomdp.jl")

export
        ShapeNode,
        CircleNode,
        EllipseNode,
        BlobNode,
        MultiShapeNode
include("shapes.jl")

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

export
    standardize,
    standardize_scale,
    calculate_standardization,
    save_standardization,
    generate_ore_mass_samples
include("standardization.jl")

export
        plot_history,
        run_trial,
        gen_cases,
        plot_ore_map,
        plot_mass_map,
        plot_volume
include("utils.jl")

export
        BeliefMDP
include("belief_mdp.jl")

end
