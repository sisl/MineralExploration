using Revise

using StatsBase
using Plots
using POMDPs
using MineralExploration

function sample_ucb_drill(mean, var)
    scores = belief_scores(mean, var)
    weights = Float64[]
    idxs = CartesianIndex{2}[]
    m, n, _ = size(mean)
    for i =1:m
        for j = 1:n
            idx = CartesianIndex(i, j)
            push!(idxs, idx)
            push!(weights, scores[idx])
        end
    end
    coords = sample(idxs, StatsBase.Weights(weights))
    # return MEAction(coords=coords)
end

function belief_scores(m, v)
    # scores = exp.((mean[:,:,1] + ucb.*sqrt.(var[:,:,1]))*t)
    # scores = mean[:,:,1] + ucb.*sqrt.(var[:,:,1])
    norm_mean = m[:,:,1]./(maximum(m[:,:,1]) - minimum(m[:,:,1]))
    norm_mean .-= minimum(norm_mean)
    norm_std = sqrt.(v[:,:,1])./(maximum(v[:,:,1]) - minimum(v[:,:,1]))
    norm_std .-= minimum(norm_std)
    scores = norm_mean.*norm_std
    # scores = norm_mean
    # scores = norm_std
    scores ./= sum(scores)
    return scores
end

# function belief_scores(m, v)
#     m_pad = zeros(Float64, size(m)[1] + 2, size(m)[2] + 2)
#     m_pad[2:size(m)[1]+1, 2:size(m)[2]+1] = m[:,:,1]
#     # m[:, :, 1]
#     grads = zeros(Float64, size(m)[1], size(m)[2])
#     for i = 2:size(m)[1] - 1
#         for j = 2:size(m)[2] - 1
#             grads[i, j] = m[i, j, 1]
#             grads[i, j] -= 0.25*m_pad[i, j]
#             grads[i, j] -= 0.25*m_pad[i+2, j]
#             grads[i, j] -= 0.25*m_pad[i, j+2]
#             grads[i, j] -= 0.25*m_pad[i+2, j+2]
#         end
#     end
#     # scores = grads./sum(grads)
#     scores = abs.(grads)
#     return scores
# end

N_INITIAL = 0
MAX_BORES = 10
UCB = 50.0
T = 100.0

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)
# s0 = rand(ds0)

up = MEBeliefUpdater(m, 5000)
println("Initializing belief...")
# b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

b = b0
# b = POMDPs.update(up, b0, MEAction(coords=CartesianIndex(20, 20)),
#                 MEObservation(s0.ore_map[20, 20, 1], false, false))
# b = POMDPs.update(up, b, MEAction(coords=CartesianIndex(30, 30)),
#                 MEObservation(s0.ore_map[30, 30, 1], false, false))

mean_ore, var_ore = MineralExploration.summarize(b)
p_sample = belief_scores(mean_ore, var_ore)
println("Plotting...")
display(plot(b))
fig = heatmap(p_sample, title="Sampling Probability, UCB=$UCB", fill=true) #, clims=(0.0, 1.0))
# savefig(fig, "./data/next_action.png")
display(fig)

x = []
y = []
for _ = 1:100
    coords = sample_ucb_drill(mean_ore, var_ore)
    push!(x, coords[1])
    push!(y, coords[2])
end

scatter!(fig, x, y, legend=:none)
display(fig)
# savefig(fig, "./data/next_action_sampled.png")
