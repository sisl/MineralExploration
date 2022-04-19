function calculate_standardization(X)
    return mean(X), std(X)
end

function standardize(X; kwargs...)
    μ, σ = calculate_standardization(X)
    return standardize(X, μ, σ; kwargs...)
end

function standardize(X, μ, σ; target_μ=0, target_σ=1)
    X′ = (X .- μ) / σ
    return target_μ .+ target_σ*X′
end

standardize_scale(x, μ, σ; kwargs...) = standardize(x, μ, σ; kwargs...) / x

function save_standardization(grid_dims, shape_types=["BlobNode", "EllipseNode", "CircleNode"];
        N=10_000, seed=0xC0FFEE, file=joinpath(@__DIR__, "standardization.json"))
    if isfile(file)
        standardization_params = open(file, "r") do f
            JSON.parse(read(f, String))
        end
    else
        standardization_params = Dict()
    end

    @info "Using standardization seed: $seed"
    Random.seed!(seed)
    mass_results, mass_params = generate_ore_mass_samples(grid_dims, shape_types; N=N, apply_scale=false)

    grid_dims_key = string(grid_dims[1], "x", grid_dims[2])
    for shape_type in shape_types
        if haskey(standardization_params, grid_dims_key)
            shapes_dict = standardization_params[grid_dims_key]
        else
            shapes_dict = Dict()
        end
        μ, σ = calculate_standardization(mass_results[shape_type])
        ShapeType = eval(Meta.parse(shape_type))
        μ, σ = standardized_adjustments(ShapeType, μ, σ, grid_dims_key)
        if haskey(shapes_dict, shape_type)
            @info "Updating $grid_dims_key standardization parameters for $shape_type"
        else
            @info "New $grid_dims_key standardization parameters for $shape_type"
            shapes_dict[shape_type] = Dict()
        end
        shapes_dict[shape_type]["mean"] = μ
        shapes_dict[shape_type]["std"] = σ
        standardization_params[grid_dims_key] = shapes_dict
    end

    open(file, "w+") do f
        JSON.print(f, JSON.parse(JSON.json(standardization_params)), 4)
    end
    @info "Wrote standardization parameters to $file"

    return standardization_params
end

# adjustments to μ and σ based on imprecise scaling due to blur and added Gaussian background noise.
standardized_adjustments(::Type{BlobNode}, μ, σ, grid_dims_key) = (μ, σ)
standardized_adjustments(::Type{CircleNode}, μ, σ, grid_dims_key) = (μ, σ)
function standardized_adjustments(::Type{EllipseNode}, μ, σ, grid_dims_key)
    if grid_dims_key == "50x50"
        return (μ+9, σ+30)
    elseif grid_dims_key == "30x30"
        return (μ+5, σ-10)
    elseif grid_dims_key == "10x10"
        return (μ+10, σ+27)
    else
        return (μ, σ)
    end
end

function generate_ore_mass_samples(grid_dims, shape_types; N=1000, apply_scale=true)
    mass_results = Dict()
    mass_params = Dict()
    for shape_type in shape_types
        ShapeType = eval(Meta.parse(shape_type))
        true_mainbody = ShapeType(grid_dims=grid_dims)
        mainbody = ShapeType(grid_dims=grid_dims)

        m = MineralExplorationPOMDP(true_mainbody_gen=true_mainbody,
                                    mainbody_gen=mainbody,
                                    grid_dim=grid_dims)
        initialize_data!(m, 0)

        masses, mass_samples, mainbody_params = generate_ore_mass_samples(m; N=N, apply_scale=apply_scale)
        mass_results[shape_type] = masses
        mass_params[shape_type] = mainbody_params
    end
    return mass_results, mass_params
end

function generate_ore_mass_samples(m; N=1000, apply_scale=true)
    ds0 = POMDPs.initialstate_distribution(m)
    samples = []
    masses = []
    params = []
    for i in 1:N
        s0 = rand(m.rng, ds0; truth=true, apply_scale=apply_scale)
        r_massive = calc_massive(s0.ore_map, m.massive_threshold, m.dim_scale)
        push!(masses, r_massive)
        push!(params, s0.mainbody_params)
        push!(samples, s0.ore_map)
    end
    return masses, samples, params
end

function get_prescaled_parameters(ShapeType::Type{<:ShapeNode}, grid_dims; file=joinpath(@__DIR__, "standardization.json"))
    if isfile(file)
        standardization_params = open(file, "r") do f
            JSON.parse(read(f, String))
        end
    else
        @warn("Standardization file does not exist ($file), generating...")
        standardization_params = save_standardization(grid_dims; file=file)
    end

    grid_dims_key = string(grid_dims[1], "x", grid_dims[2])
    if !haskey(standardization_params, grid_dims_key)
        standardization_params = save_standardization(grid_dims; file=file)
    end

    shape_type = string(ShapeType.name.name)
    if !haskey(standardization_params[grid_dims_key], shape_type)
        standardization_params = save_standardization(grid_dims, [shape_type]; file=file)
    end

    params = standardization_params[grid_dims_key][shape_type]
    μ, σ = params["mean"], params["std"]
    return μ, σ
end
