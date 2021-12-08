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
    abs_errs = Float64[abs(mean_vols - r_massive)]
    vol_stds = Float64[std_vols]
    dists = Float64[]
    println("Entering Simulation...")
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=50)
        @show t
        @show a.type
        @show a.coords
        discounted_return += POMDPs.discount(m)^(t - 1)*r
        dist = sqrt(sum(([a.coords[1], a.coords[2]] .- 25.0).^2)) #TODO only for single fixed

        b_fig = plot(bp, t)
        vols = [sum(p.ore_map .>= m.massive_threshold) for p in bp.particles]
        mean_vols = mean(vols)
        std_vols = std(vols)
        println("Vols: $mean_vols ± $std_vols")

        if a.type == :drill
            push!(dists, dist)
            push!(abs_errs, abs(mean_vols - r_massive))
            push!(vol_stds, std_vols)

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
    end
    println("Discounted Return: $discounted_return")
    dist_fig = plot(dists)
    abs_err_fig = plot(abs_errs)
    vols_fig = plot(vol_stds./vol_stds[1])
    if isa(save_dir, String)
        path = string(save_dir, "dists.png")
        savefig(dist_fig, path)

        path = string(save_dir, "abs_err.png")
        savefig(abs_err_fig, path)

        path = string(save_dir, "vol_std.png")
        savefig(vols_fig, path)
    end
    if display_figs
        display(dist_fig)
        display(abs_err_fig)
        display(vols_fig)
    end
    return (discounted_return, dists, abs_errs, vol_stds)
end
