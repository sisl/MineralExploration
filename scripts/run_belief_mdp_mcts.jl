using Revise

import BSON: @save, @load
using D3Trees
using POMDPs
using POMCPOW
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using Statistics
using Images
using Random
using Infiltrator
using MineralExploration
using POMDPPolicies
using MCTS
using MixedFidelityModelSelection

function MCTS.node_tag(s::MEBelief)
    return "b($(s.stopped), $(s.decided))"
end

# Random.seed!(1000) # determinism (truth: ~99 ore volume)
Random.seed!(5000) # determinism (truth: ~245 ore volume)

N_INITIAL = 0
MAX_BORES = 25
MIN_BORES = 5
GRID_SPACING = 0
MAX_MOVEMENT = 20

SAVE_DIR = "C:/Users/mossr/Code/sisl/CCS/MineralExploration/scripts/data/demos/bmdp_mcts"

!isdir(SAVE_DIR) && mkdir(SAVE_DIR)

grid_dims = (30, 30, 1)
true_mainbody = BlobNode(grid_dims=(50,50,1), factor=4)
mainbody = BlobNode(grid_dims=grid_dims)


m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                            true_mainbody_gen=true_mainbody, mainbody_gen=mainbody, original_max_movement=MAX_MOVEMENT,
                            min_bores=MIN_BORES, grid_dim=grid_dims)
initialize_data!(m, N_INITIAL)
@show m.max_movement

ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0; truth=true)

Random.seed!(52992) # restart determinism

up = MEBeliefUpdater(m, 1000, 2.0)
b0 = POMDPs.initialize_belief(up, ds0)

next_action = NextActionSampler()
f_next_action(bmdp::BeliefMDP, b::MEBelief, h) = POMCPOW.next_action(next_action, bmdp.pomdp, b, h)
exploration_coefficient = 100.0
k_action = 2.0
alpha_action = 0.25
tree_queries = [100, 1_000, 10_000]
i_tree_queries = 1

usepomcpow = false
if usepomcpow
    solver = POMCPOWSolver(tree_queries=tree_queries[i_tree_queries],
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        next_action=next_action,
                        k_action=k_action,
                        alpha_action=alpha_action,
                        k_observation=2.0,
                        alpha_observation=0.1,
                        criterion=POMCPOW.MaxUCB(exploration_coefficient),
                        final_criterion=POMCPOW.MaxQ(),
                        # final_criterion=POMCPOW.MaxTries(),
                        estimate_value=0.0, # (pomdp, s, d) -> MineralExploration.extraction_reward(pomdp, s, ......)
                        # estimate_value=leaf_estimation,
                        )
    planner = POMDPs.solve(solver, m)
else
	# belief_reward(pomdp::POMDP, b, a, bp) = mean(reward(pomdp, sp, a) - reward(pomdp, s, a) for (s,sp) in zip(particles(b), particles(bp)))
	belief_reward(pomdp::POMDP, b, a, bp) = mean(reward(pomdp, s, a) for s in particles(b))
	bmdp = BeliefMDP(m, up, belief_reward)
    mcts_iterations = 10 # (10 takes ~80 seconds, 100 takes 10 minutes, 1000 takes 2.5 hours)
    mcts_c = exploration_coefficient
    estimate_value = (bmdp, b, d) -> 0.0 # mean(MineralExploration.extraction_reward(bmdp.pomdp, s) for s in particles(b))
	solver = DPWSolver(n_iterations=mcts_iterations,
                       check_repeat_action=true,
                       exploration_constant=mcts_c,
                       next_action=f_next_action,
                       k_action=k_action,
                       alpha_action=alpha_action,
                       tree_in_info=true,
                       estimate_value=estimate_value,
                       show_progress=true)
	planner = solve(solver, bmdp)
end

timing = @timed results = run_trial(m, up, planner, s0, b0, save_dir=SAVE_DIR, display_figs=true, collect_training_data=true)
@show timing.time

@show planner.tree.q

showtree(tree) = inchrome(D3Tree(tree))
