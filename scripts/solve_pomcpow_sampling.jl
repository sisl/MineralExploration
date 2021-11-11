try using Pkg catch;Pkg.add("Pkg"); using Pkg end
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end

try using ArgParse catch;Pkg.add("ArgParse"); using ArgParse end


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

# Hyper Parameters

index_from_script = parse(Int64, ARGS[1])


function look_up_table(a_list)

    count_list = []
    combination_list = []
    count_list = [length(ele) for ele in a_list]
    combination_list = [push!(combination_list,1:i) for i in count_list][1]
    total_row = prod(count_list)
    reference_table = reshape(collect(Iterators.product(combination_list...)),total_row,1)
    return reference_table
end

function get_hyperParameters(index_from_script)
    K_choices = [1, 2, 4]
    alpha_choices = [0.0, 1.1, 2.2]
    UCB_choices = [1.2, 1.3, 1.4]
    leaf_est_choices= [leaf_estimation, 0.0]
    varible_options = [K_choices,alpha_choices,UCB_choices,leaf_est_choices]

    reference = look_up_table(varible_options)
    reference_index = reference[index_from_script]
    return_values = []
    for i in 1:length(reference_index)
        push!(return_values,varible_options[i][reference_index[i]])

    end


    return return_values
end

hyper_params = get_hyperParameters(index_from_script)



N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

up = MEBeliefUpdater(m, 100)
println("Initializing belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")
next_action = NextActionSampler() #b0, up)

solver = POMCPOWSolver(tree_queries=1000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=hyper_params[1],
                       alpha_action=hyper_params[2],
                       k_observation=2,
                       alpha_observation=0.1,
                       criterion=POMCPOW.MaxUCB(hyper_params[3]),
                       final_criterion=POMCPOW.MaxQ(),
                       # final_criterion=POMCPOW.MaxTries(),
                       # estimate_value=0.0
                       estimate_value=hyper_params[4]
                       )
planner = POMDPs.solve(solver, m)

# @profview POMCPOW.action_info(planner, b0, tree_in_info=true)
# @profview POMCPOW.action_info(planner, b0, tree_in_info=true)
# volumes = [sum(b.ore_map[:,:,1] .>= m.massive_threshold) for b in b0.particles]
# mean(volumes)
# MineralExploration.std(volumes)

# println("Building test tree...")
# a, info = POMCPOW.action_info(planner, B[8], tree_in_info=true)
# tree = info[:tree]
# inbrowser(D3Tree(tree, init_expand=1), "firefox")

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

vars = [p.var for p in b0.particles]
mean_vars = mean(vars)
std_vars = std(vars)
println("Vars: $mean_vars ± $std_vars")
# fig = histogram(vars, bins=10 )
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
    a_new = a
    b_new = bp
    @show t
    @show a
    @show r
    @show sp.stopped
    @show bp.stopped

    volumes = Float64[sum(p.ore_map[:,:,1] .>= m.massive_threshold) for p in bp.particles]
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

    vars = [p.var for p in bp.particles]
    mean_vars = mean(vars)
    std_vars = std(vars)
    @show mean_vars
    @show std_vars
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
