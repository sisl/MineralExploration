
using MineralExploration

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics

using MineralExploration

N_SIM = 4
N_PROCS = 2
N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)

up = MEBeliefUpdater(m, 100)
println("Initializing Belief...")
# b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

next_action = NextActionSampler()

solver = POMCPOWSolver(tree_queries=100,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=3,
                       alpha_action=0.25,
                       k_observation=2,
                       alpha_observation=0.25,
                       criterion=POMCPOW.MaxUCB(10.0),
                       estimate_value=POMCPOW.RolloutEstimator(ExpertPolicy(m))
                       )
planner = POMDPs.solve(solver, m)

println("Building Simulation Queue...")
queue = POMDPSimulators.Sim[]
for i = 1:N_SIM
    s0 = rand(ds0)
    s_massive = s0.ore_map[:,:,1] .>= 0.7
    r_massive = sum(s_massive)
    push!(queue, POMDPSimulators.Sim(m, planner, up, b0, s0, metadata=Dict(:massive_ore=>r_massive)))
end
println("Building Workers $N_PROCS...")
POMDPSimulators.addprocs(N_PROCS)
println("Starting Simulations...")
data = POMDPSimulators.run_parallel(queue, show_progress=false)
println("Simulations Complete!")
