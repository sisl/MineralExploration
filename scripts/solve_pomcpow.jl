using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots

using CCS

DIMS = (80, 80, 1)

N_INITIAL = 3
MAX_WELLS = 3

spec = POMDPSpecification(grid_dim=DIMS, max_n_wells=MAX_WELLS)
initialize_data!(spec, N_INITIAL)
m = TestPOMDP2D(spec)
ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

up = ReservoirBeliefUpdater(spec)
b0 = initialize_belief(up, s0)

solver = POMCPOWSolver(tree_queries=100, check_repeat_obs=false, check_repeat_act=true)
planner = POMDPs.solve(solver, m)

fig = heatmap(s0.porosity[:,:,1], title="True Porosity Field", fill=true, clims=(0.1, 0.6))
# display(fig)
# println("Entering Simulation...")
# for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=50)
#     @show a
#     @show r
#     @show t
#     # fig = plot(bp)
#     # display(fig)
# end
