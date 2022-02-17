using Revise

import BSON: @save, @load
using POMDPs
using POMCPOW
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using Statistics
using Images
using Random
Random.seed!(1004) # determinism

using MineralExploration

N_INITIAL = 0
MAX_BORES = 25
MIN_BORES = 5
GRID_SPACING = 0
MAX_MOVEMENT = 20
SAVE_DIR = "./data/demos/multishape_sandbox/"
!isdir(SAVE_DIR) && mkdir(SAVE_DIR)

grid_dims = (50, 50, 1)
true_mainbody = BlobNode(grid_dims=grid_dims, factor=4)
mainbody = BlobNode(grid_dims=grid_dims)
# mainbody1 = CircleNode(grid_dims=grid_dims)
# mainbody = MultiShapeNode([mainbody1, mainbody2])
# mainbody = MultiShapeNode([mainbody1])

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                            true_mainbody_gen=true_mainbody, mainbody_gen=mainbody, original_max_movement=MAX_MOVEMENT,
                            min_bores=MIN_BORES, grid_dim=grid_dims)
initialize_data!(m, N_INITIAL)
@show m.max_movement

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0; truth=true)

Random.seed!(1002) # determinism

# s0.ore_map[:,:,1] = downscale(s0.ore_map[:,:,1], (10,10))
# s0.ore_map[:] = reshape(imresize(s0.ore_map[:,:,1], (10,10)), (10,10,1)) # TODO ?

up = MEBeliefUpdater(m, 1000, 2.0)
b0 = POMDPs.initialize_belief(up, ds0)

next_action = NextActionSampler()
tree_queries = [100, 1_000, 10_000]
i_tree_queries = 1
solver = POMCPOWSolver(tree_queries=tree_queries[i_tree_queries],
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=2.0,
                       alpha_action=0.25,
                       k_observation=2.0,
                       alpha_observation=0.1,
                       criterion=POMCPOW.MaxUCB(100.0),
                       final_criterion=POMCPOW.MaxQ(),
                       # final_criterion=POMCPOW.MaxTries(),
                       estimate_value=0.0
                       # estimate_value=leaf_estimation
                       )
planner = POMDPs.solve(solver, m)

timing = @timed results = run_trial(m, up, planner, s0, b0, save_dir=SAVE_DIR, display_figs=false)
@show timing.time