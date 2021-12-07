function plot_error_history(hs::Vector, n_max::Int64=10)
    μ = Float64[]
    σ = Float64[]
    for i = 1:n_max
        vals = Float64[]
        for h in hs
            if length(h) >= i
                push!(vals, h[i])
            end
        end
        push!(μ, mean(vals))
        push!(σ, std(vals))
    end
    σ .*= 1.0 .- isnan.(σ)
    return (plot(μ, yerror=σ), μ, σ)
end

function run_trial(m::MineralExplorationPOMDP, up::MEBeliefUpdater, policy::POMDPs.Policy;
                display_figs::Bool=true, save_dir::Union{Nothing, String}=nothing)
    ds0 = POMDPs.initialstate_distribution(m)
    s0 = rand(ds0)
    println("Initializing belief...")
    b0 = POMDPs.initialize_belief(up, ds0)
    println("Belief Initialized!")

    ore_fig = heatmap(s0.ore_map[:,:,1], title="True Ore Field", fill=true, clims=(0.0, 1.0))
    s_massive = s0.ore_map .>= m.massive_threshold
    r_massive = sum(s_massive)
    mass_fig = heatmap(s_massive[:,:,1], title="Massive Ore Deposits: $r_massive", fill=true, clims=(0.0, 1.0))
    b0_fig = plot(b0)

    vols = [sum(p.ore_map .>= m.massive_threshold) for p in b0.particles]
    mean_vols = mean(vols)
    std_vols = std(vols)
    println("Vols: $mean_vols ± $std_vols")

    h = fit(Histogram, vols, [0:25:300;])
    h = normalize(h, mode=:probability)
    # b0_hist = histogram(vols, title="Belief Volumes t=0", bins=[0:25:300;],
    #                                     normalize=:probability, legend=:none)
    b0_hist = plot(h, title="Belief Volumes t=0", legend=:none)
    plot!(b0_hist, [r_massive, r_massive], [0.0, maximum(h.weights)], linecolor=:red, linewidth=3)
    if isa(save_dir, String)
        path = string(save_dir, "ore_map.png")
        savefig(ore_fig, path)

        path = string(save_dir, "mass_map.png")
        savefig(mass_fig, path)

        path = string(save_dir, "b0.png")
        savefig(b0_fig, path)

        path = string(save_dir, "b0_hist.png")
        savefig(b0_hist, path)
    end
    if display_figs
        display(ore_fig)
        display(mass_fig)
        display(b0_fig)
        display(b0_hist)
    end


    discounted_return = 0.0
    println("Entering Simulation...")
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=50)
        @show t
        @show a.type
        @show a.coords
        discounted_return += POMDPs.discount(m)^(t - 1)*r

        b_fig = plot(bp, t)
        vols = [sum(p.ore_map .>= m.massive_threshold) for p in bp.particles]
        mean_vols = mean(vols)
        std_vols = std(vols)
        println("Vols: $mean_vols ± $std_vols")

        # b_hist = histogram(vols, title="Belief Volumes t=$t", bins=[0:25:300;],
        #                                     normalize=:probability, legend=:none)
        h = fit(Histogram, vols, [0:25:300;])
        h = normalize(h, mode=:probability)
        b_hist = plot(h, title="Belief Volumes t=$t", legend=:none)
        plot!(b_hist, [r_massive, r_massive], [0.0, maximum(h.weights)], linecolor=:red, linewidth=3)
        if isa(save_dir, String)
            path = string(save_dir, "b$t.png")
            savefig(b_fig, path)

            path = string(save_dir, "b$(t)_hist.png")
            savefig(b_hist, path)
        end
        if display_figs
            display(b_fig)
            display(b_hist)
        end
    end
    println("Discounted Return: $discounted_return")
end
