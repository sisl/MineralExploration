mutable struct BetaZeroTrainingData
    b # current belief
    π # current policy estimate (using N(s,a))
    z # final outcome (discounted return of the episode)
end

function plot_history(hs::Vector, n_max::Int64=10,
                        title::Union{Nothing, String}=nothing,
                        y_label::Union{Nothing, String}=nothing,
                        box_plot::Bool=false)
    μ = Float64[]
    σ = Float64[]
    vals_vector = Vector{Float64}[]
    for i = 1:n_max
        vals = Float64[]
        for h in hs
            if length(h) >= i
                push!(vals, h[i])
            end
        end
        push!(μ, mean(vals))
        push!(σ, std(vals)/sqrt(length(vals)))
        push!(vals_vector, vals)
    end
    σ .*= 1.0 .- isnan.(σ)
    if isa(title, String)
        if box_plot
            fig = plot(μ, legend=:none, title=title, ylabel=y_label)
            for (i, vals) in enumerate(vals_vector)
                boxplot!(fig, repeat([i], outer=length(vals)), vals, color=:white, outliers=false)
            end
        else
            fig = plot(μ, yerror=σ, legend=:none, title=title, ylabel=y_label)
        end
    else
        if box_plot
            fig = plot(μ, legend=:none)
            for (i, vals) in enumerate(vals_vector)
                boxplot!(fig, i, vals, color=:white, outliers=false)
            end
        else
            fig = plot(μ, yerror=σ, legend=:none)
        end
    end
    return (fig, μ, σ)
end

function gen_cases(ds0::MEInitStateDist, n::Int64, save_dir::Union{String, Nothing}=nothing)
    states = MEState[]
    for i = 1:n
        push!(states, rand(ds0.rng, ds0))
    end
    if isa(save_dir, String)
        save(save_dir, "states", states)
    end
    return states
end

function run_trial(m::MineralExplorationPOMDP, up::POMDPs.Updater,
                policy::POMDPs.Policy, s0::MEState, b0::MEBelief;
                display_figs::Bool=true, save_dir::Union{Nothing, String}=nothing,
                return_final_belief=false, return_all_trees=false, collect_training_data=false,
                cmap=:viridis, verbose::Bool=true)
    if verbose
        println("Initializing belief...")
    end
    if verbose
        println("Belief Initialized!")
    end

    ore_fig = plot_ore_map(s0.ore_map, cmap)
    mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, cmap; truth=true)
    b0_fig = plot(b0; cmap=cmap)
    b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=verbose)

    if isa(save_dir, String)
        save_dir = joinpath(abspath(save_dir), "")

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

    b_mean, b_std = MineralExploration.summarize(b0)
    if isa(save_dir, String)
        path = string(save_dir, "belief_mean.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_mean, :, 1))
        end
        path = string(save_dir, "belief_std.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_std, :, 1))
        end
    end

    last_action = :drill
    n_drills = 0
    discounted_return = 0.0
    ae = mean(abs.(vols .- r_massive))
    re = mean(vols .- r_massive)
    abs_errs = Float64[ae]
    rel_errs = Float64[re]
    vol_stds = Float64[std_vols]
    dists = Float64[]
    final_belief = nothing
    trees = []
    if verbose
        println("Entering Simulation...")
    end
    if collect_training_data
        training_data = [BetaZeroTrainingData(get_input_representation(b0), nothing, nothing)]
    end
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=m.max_bores+2, rng=m.rng)
        discounted_return += POMDPs.discount(m)^(t - 1)*r
        dist = sqrt(sum(([a.coords[1], a.coords[2]] .- 25.0).^2)) #TODO only for single fixed
        last_action = a.type

        b_fig = plot(bp, t; cmap=cmap)
        b_hist, vols, mean_vols, std_vols = plot_volume(m, bp, r_massive; t=t, verbose=false)

        if verbose
            @show t
            @show a.type
            @show a.coords
            println("Vols: $mean_vols ± $std_vols")
        end

        if a.type == :drill
            n_drills += 1
            ae = mean(abs.(vols .- r_massive))
            re = mean(vols .- r_massive)
            push!(dists, dist)
            push!(abs_errs, ae)
            push!(rel_errs, re)
            push!(vol_stds, std_vols)

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
            b_mean, b_std = MineralExploration.summarize(bp)
            if isa(save_dir, String)
                path = string(save_dir, "belief_mean.txt")
                open(path, "a") do io
                    writedlm(io, reshape(b_mean, :, 1))
                end
                path = string(save_dir, "belief_std.txt")
                open(path, "a") do io
                    writedlm(io, reshape(b_std, :, 1))
                end
            end
        end
        final_belief = bp
        if return_all_trees
            push!(trees, deepcopy(policy.tree))
        end
        if collect_training_data && a.type == :drill # NOTE that :stop and beyond have the same belief representation and obs.
            data = BetaZeroTrainingData(get_input_representation(bp), nothing, nothing)
            push!(training_data, data)
        end
    end
    if verbose
        println("Discounted Return: $discounted_return")
    end
    if collect_training_data
        for d in training_data
            d.z = discounted_return
        end
    end
    ts = [1:length(abs_errs);] .- 1
    dist_fig = plot(ts[2:end], dists, title="bore distance to center",
                    xlabel="time step", ylabel="distance", legend=:none, lw=2, c=:crimson)

    abs_err_fig = plot(ts, abs_errs, title="absolute volume error",
                    xlabel="time step", ylabel="absolute error", legend=:none, lw=2, c=:crimson)

    rel_err_fig = plot(ts, rel_errs, title="relative volume error",
                    xlabel="time step", ylabel="relative error", legend=:none, lw=2, c=:crimson)
    rel_err_fig = plot!([xlims()...], [0, 0], lw=1, c=:black, xlims=xlims())

    vols_fig = plot(ts, vol_stds./vol_stds[1], title="volume standard deviation",
                    xlabel="time step", ylabel="standard deviation", legend=:none, lw=2, c=:crimson)
    if isa(save_dir, String)
        path = string(save_dir, "dists.png")
        savefig(dist_fig, path)

        path = string(save_dir, "abs_err.png")
        savefig(abs_err_fig, path)

        path = string(save_dir, "rel_err.png")
        savefig(rel_err_fig, path)

        path = string(save_dir, "vol_std.png")
        savefig(vols_fig, path)
    end
    if display_figs
        display(dist_fig)
        display(abs_err_fig)
        display(rel_err_fig)
        display(vols_fig)
    end
    return_values = (discounted_return, dists, abs_errs, rel_errs, vol_stds, n_drills, r_massive, last_action)
    if return_final_belief
        return_values = (return_values..., final_belief)
    end
    if return_all_trees
        return_values = (return_values..., trees)
    end
    if collect_training_data
        return_values = (return_values..., training_data)
    end
    return return_values
end

function plot_ore_map(ore_map, cmap=:viridis)
    xl = (0.5, size(ore_map,1)+0.5)
    yl = (0.5, size(ore_map,2)+0.5)
    return heatmap(ore_map[:,:,1], title="true ore field", fill=true, clims=(0.0, 1.0), aspect_ratio=1, xlims=xl, ylims=yl, c=cmap)
end

function plot_mass_map(ore_map, massive_threshold, cmap=:viridis; dim_scale=1, truth=false)
    xl = (0.5, size(ore_map,1)+0.5)
    yl = (0.5, size(ore_map,2)+0.5)
    s_massive = ore_map .>= massive_threshold
    r_massive = dim_scale*sum(s_massive)
    mass_fig = heatmap(s_massive[:,:,1], title="massive ore deposits: $(round(r_massive, digits=2))", fill=true, clims=(0.0, 1.0), aspect_ratio=1, xlims=xl, ylims=yl, c=cmap)
    return (mass_fig, r_massive)
end

function plot_volume(m::MineralExplorationPOMDP, b0::MEBelief, r_massive::Real; t=0, verbose::Bool=true)
    vols = [m.dim_scale*sum(p.ore_map .>= m.massive_threshold) for p in b0.particles]
    mean_vols = round(mean(vols), digits=2)
    std_vols = round(std(vols), digits=2)
    if verbose
        println("Vols: $mean_vols ± $std_vols")
    end
    h = fit(Histogram, vols, [0:10:400;])
    h = normalize(h, mode=:probability)

    b0_hist = plot(h, title="belief volumes t=$t, μ=$mean_vols, σ=$std_vols", legend=:none, c=:cadetblue)
    h_height = maximum(h.weights)
    plot!(b0_hist, [r_massive, r_massive], [0.0, h_height], linecolor=:crimson, linewidth=3)
    plot!([mean_vols, mean_vols], [0.0, h_height], linecolor=:crimson, linestyle=:dash, linewidth=2, label=false)
    plot!([m.extraction_cost, m.extraction_cost], [0.0, h_height/3], linecolor=:gold, linewidth=4, label=false)
    ylims!(0, h_height*1.05)

    return (b0_hist, vols, mean_vols, std_vols)
end
