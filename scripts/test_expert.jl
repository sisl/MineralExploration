using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)

up = MEBeliefUpdater(m, 1000)
b0 = POMDPs.initialize_belief(up, ds0)

policy = ExpertPolicy(m)

N = 100
rs = RolloutSimulator(max_steps=10)
V = Float64[]
println("Starting simulations")
for i in 1:N
    if (i%1) == 0
        println("Trial $i")
    end
    s0 = rand(ds0)
    v = simulate(rs, m, policy, up, b0, s0)
    push!(V, v)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")
