using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics

using ProfileView
using D3Trees

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=4)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

up = MEBeliefUpdater(m, 10000)
println("Initializing belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

next_action = NextActionSampler() #b0, up)
solver = POMCPOWSolver(tree_queries=1000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=3,
                       alpha_action=0.25,
                       k_observation=2,
                       alpha_observation=0.25,
                       criterion=POMCPOW.MaxUCB(100.0),
                       final_criterion=POMCPOW.MaxQ(),
                       # final_criterion=POMCPOW.MaxTries(),
                       # estimate_value=0.0
                       estimate_value=leaf_estimation
                       )
planner = POMDPs.solve(solver, m)

# @profview POMCPOW.action_info(planner, b0, tree_in_info=true)
# @profview POMCPOW.action_info(planner, b0, tree_in_info=true)
# volumes = [sum(b.ore_map[:,:,1] .>= m.massive_threshold) for b in b0.particles]
# mean(volumes)
# MineralExploration.std(volumes)

println("Building test tree...")
a, info = POMCPOW.action_info(planner, b0, tree_in_info=true)
tree = info[:tree]
inbrowser(D3Tree(tree, init_expand=1), "firefox")

println("Plotting...")
fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
# savefig(fig, "./data/example/ore_vals.png")
display(fig)

s_massive = s0.ore_map[:,:,1] .>= 0.7
r_massive = sum(s_massive)
println("Massive ore: $r_massive")
println("MB Variance: $(s0.var)")

fig = heatmap(s_massive, title="Massive Ore Deposits: $r_massive", fill=true, clims=(0.0, 1.0))
# savefig(fig, "./data/example/massive.png")
display(fig)

fig = plot(b0)
display(fig)

vars = [p[1] for p in b0.particles]
mean_vars = mean(vars)
std_vars = std(vars)
println("Vars: $mean_vars ± $std_vars")

vols = [sum(p[2] .>= 0.7) for p in b0.particles]
mean_vols = mean(vols)
std_vols = std(vols)
println("Vols: $mean_vols ± $std_vols")
# fig = histogram(vars, bins=10 )
# display(fig)
# fig = histogram(vols, bins=10 )
# display(fig)
b_new = nothing
a_new = nothing
discounted_return = 0.0
B = [b0]
println("Entering Simulation...")
for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", max_steps=50)
    global discounted_return
    global b_new
    global a_new
    local fig
    local volumes
    local mb_var

    local vars
    local mean_vars
    local std_vars
    a_new = a
    b_new = bp
    @show t
    @show a
    @show r
    @show sp.stopped
    @show bp.stopped

    volumes = Float64[sum(p[2][:,:,1] .>= m.massive_threshold) for p in bp.particles]
    mean_volume = mean(volumes)
    std_volume = std(volumes)
    volume_lcb = mean_volume - 1.0*std_volume
    push!(B, bp)
    @show mean_volume
    @show std_volume
    @show volume_lcb

    fig = plot(bp, t)
    str = "./data/example/belief_$t.png"
    # savefig(fig, str)
    display(fig)

    vars = [p[1] for p in bp.particles]
    mean_vars = mean(vars)
    std_vars = std(vars)
    println("Vars: $mean_vars ± $std_vars")
    # fig = histogram(vars, bins=10)
    # display(fig)
    discounted_return += POMDPs.discount(m)^(t - 1)*r
end

println("Decision: $(a_new.type)")
println("Massive Ore: $r_massive")
println("Mining Profit: $(r_massive - m.extraction_cost)")
println("Episode Return: $discounted_return")

# m, v = MineralExploration.summarize(b_new)
# scores = MineralExploration.belief_scores(m, v)
# display(heatmap(scores))
# plot(b_new)
