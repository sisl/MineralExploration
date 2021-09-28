using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots

using MineralExploration

DIMS = (80, 80, 1)

N_INITIAL = 3
MAX_BORES = 5

m = MineralExplorationPOMDP(grid_dim=DIMS, max_bores=MAX_BORES)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

# up = ReservoirBeliefUpdater(spec)
# b0 = initialize_belief(up, s0)
#
# solver = POMCPOWSolver(tree_queries=100, check_repeat_obs=false, check_repeat_act=true)
# planner = POMDPs.solve(solver, m)
#
fig = heatmap(s0.ore_map[:,:,1], title="True Porosity Field", fill=true, clims=(0.0, 1.0))
savefig(fig, "mineral.png")
display(fig)

# # println("Entering Simulation...")
# # for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=50)
# #     @show a
# #     @show r
# #     @show t
# #     # fig = plot(bp)
# #     # display(fig)
# # end
