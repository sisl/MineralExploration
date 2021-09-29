using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

up = MEBeliefUpdater(m, 100)
b0 = POMDPs.initialize_belief(up, ds0)

solver = POMCPOWSolver(tree_queries=1000, check_repeat_obs=false, check_repeat_act=true)
planner = POMDPs.solve(solver, m)

fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
display(fig)

s_massive = s0.ore_map[:,:,1] .>= 0.7

fig = heatmap(s_massive, title="Massive Ore Deposits", fill=true, clims=(0.0, 1.0))
display(fig)

fig = plot(b0)
display(fig)

discounted_return = 0.0
println("Entering Simulation...")
for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=50)
    global discounted_return
    @show a
    @show r
    @show t
    fig = plot(bp)
    display(fig)
    discounted_return += POMDPs.discount(m)^(t - 1)*r
end
println("Episode Return: $discounted_return")
