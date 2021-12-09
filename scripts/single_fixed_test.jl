using Revise

using POMDPs
using POMCPOW
using Plots
using Statistics
using StatsBase

using MineralExploration

N = 100
N_INITIAL = 0
MAX_BORES = 20
GRID_SPACING = 1
MAX_MOVEMENT = 0

mainbody = SingleFixedNode()
m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                            mainbody_gen=mainbody, max_movement=MAX_MOVEMENT)
initialize_data!(m, N_INITIAL)

up = MEBeliefUpdater(m, 1000, 2.0)

next_action = NextActionSampler()
solver = POMCPOWSolver(tree_queries=1000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=2.0,
                       alpha_action=0.25,
                       k_observation=2.0,
                       alpha_observation=0.1,
                       criterion=POMCPOW.MaxUCB(100.0),
                       final_criterion=POMCPOW.MaxQ(),
                       estimate_value=0.0
                       )
planner = POMDPs.solve(solver, m)

returns = Float64[]
ores = Float64[]
decisions = Symbol[]
distances = Vector{Float64}[]
abs_errs = Vector{Float64}[]
vol_stds = Vector{Float64}[]
n_drills = Int64[]
for i = 1:N
    println("Running Trial $i")
    results = run_trial(m, up, planner, save_dir=nothing, display_figs=false, verbose=false)
    push!(returns, results[1])
    push!(ores, results[6])
    push!(decisions, results[7])
    push!(distances, results[2])
    push!(abs_errs, results[3])
    push!(vol_stds, results[4])
    push!(n_drills, results[5])
end

fig, mu, sig = plot_history(distances, 10, "Distance to Center", "Distance")
savefig(fig, "./data/single_fixed_test/distances.png")
display(fig)

fig, mu, sig = plot_history(abs_errs, 10, "Mean Absolute Errors", "MAE")
savefig(fig, "./data/single_fixed_test/mae.png")
display(fig)


for vol_std in vol_stds
    vol_std ./= vol_std[1]
end
fig, mu, sig = plot_history(vol_stds, 10, "Volume Standard Deviation Ratio", "σ/σ₀")
savefig(fig, "./data/single_fixed_test/vol_stds.png")
display(fig)

h = fit(Histogram, n_drills)
h = StatsBase.normalize(h, mode=:probability)
b_hist = plot(h, title="Number of Bores", legend=:none)
savefig(b_hist, "./data/single_fixed_test/bore_hist.png")
display(b_hist)

abandoned = [a == :abandon for a in decisions]
mined = [a == :mine for a in decisions]

profitable = ores .> m.extraction_cost
lossy = ores .<= m.extraction_cost

n_profitable = sum(profitable)
n_lossy = sum(lossy)

profitable_mined = sum(mined.*profitable)
profitable_abandoned = sum(abandoned.*profitable)

lossy_mined = sum(mined.*lossy)
lossy_abandoned = sum(abandoned.*lossy)

mined_profit = sum(mined.*(ores .- m.extraction_cost))
available_profit = sum(profitable.*(ores .- m.extraction_cost))

# mean_drills = mean(D)
# mined_drills = sum(D.*mined)/sum(mined)
# abandoned_drills = sum(D.*abandoned)/sum(abandoned)

println("Available Profit: $available_profit, Mined Profit: $mined_profit")
println("Profitable: $(sum(profitable)), Mined: $profitable_mined, Abandoned: $profitable_abandoned")
println("Lossy: $(sum(lossy)), Mined: $lossy_mined, Abandoned: $lossy_abandoned")
# println("Mean Bores: $mean_drills, Mined Bores: $mined_drills, Abandon Bores: $abandoned_drills")

h = fit(Histogram, ores[mined] .- m.extraction_cost, [-20:10:100;])
# h = StatsBase.normalize(h, mode=:probability)
ore_hist = plot(h, title="Mined Profit", legend=:none)
savefig(ore_hist, "./data/single_fixed_test/mined_ore_hist.png")
display(ore_hist)

mean_drills = mean(n_drills)
mined_drills = sum(n_drills.*mined)/sum(mined)
abandoned_drills = sum(n_drills.*abandoned)/sum(abandoned)
println("Mean Bores: $mean_drills, Mined Bores: $mined_drills, Abandon Bores: $abandoned_drills")
