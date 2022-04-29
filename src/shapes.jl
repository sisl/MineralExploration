## Shape-based mainbody nodes
abstract type ShapeNode <: MainbodyGen end

σ_blur(grid_dims) = grid_dims[1]/25 # to be σ=2 when grid is 50x50

function clamp2dims!(x, dims)
    x[1] = clamp(x[1], 1, dims[1])
    x[2] = clamp(x[2], 1, dims[2])
    return x
end

function generate_shape_matrix(shape::ShapeNode; σ=σ_blur(shape.grid_dims), grayscale=0.4)
    grid_dims = shape.grid_dims
    mat = @imagematrix begin
        gsave()
        background("black")
        Luxor.origin(Luxor.Point(0,0))
        pos = Luxor.Point(shape.center...)
        Luxor.translate(pos)
        if hasproperty(shape, :angle)
            Luxor.rotate(shape.angle)
        end
        sethue(grayscale, grayscale, grayscale)
        draw(shape)
        grestore()
    end grid_dims[1] grid_dims[2]

    return convert(Matrix{Float64}, Gray.(blur(mat, σ)))
end

Base.convert(::Type{Float64}, m::ColorTypes.ARGB32) = convert(Float64, Gray(m))

blur(img, σ) = imfilter(img, Kernel.gaussian(σ))

function scale_sample(d::MEInitStateDist, mainbody::ShapeNode, lode_map, gp_ore_map, lode_params; target_μ, target_σ)
    grid_dims = size(lode_map)
    μ, σ = get_prescaled_parameters(typeof(mainbody), grid_dims)

    ore_map = lode_map + gp_ore_map
    truth = size(ore_map) == d.true_gp_distribution.grid_dims
    dim_scale = truth ? 1 : d.dim_scale
    r_massive_prescale = calc_massive(ore_map, d.massive_thresh, dim_scale)
    scale = standardize_scale(r_massive_prescale, μ, σ; target_μ=target_μ, target_σ=target_σ)
    lode_map, lode_params = scale_sample(mainbody, lode_params, scale)

    lode_map = normalize_and_weight(lode_map, d.mainbody_weight)
    ore_map = lode_map + gp_ore_map

    return (ore_map, lode_params)
end

function scale_sample(d::MEInitStateDist, mainbody::MainbodyGen, lode_map, gp_ore_map, lode_params; kwargs...)
    # pass through for SingleFixedNode, SingleVarNode, and MultiVarNode
    ore_map = lode_map + gp_ore_map
    return ore_map, lode_params
end

## Circular Shaped Node
"""
Distribution for determining the center of the mainbody.
"""
function center_distribution(grid_dims; bounds=[grid_dims[1]/4, grid_dims[1]/2])
    xdistr = Distributions.Uniform(bounds[1], bounds[2])
    ydistr = Distributions.Uniform(bounds[1], bounds[2])
    return Product([xdistr, ydistr])
end

"""
Mainbody shape parameterized as a circle with `center` and `radius`.
"""
@with_kw struct CircleNode <: ShapeNode
    grid_dims::Tuple{Real, Real, Real}
    center::Union{Distribution, Vector} = center_distribution(grid_dims)
    radius::Union{Distribution, Real} = Distributions.Uniform(grid_dims[1]/7, grid_dims[1]/5)
end

"""
Draw circle using Luxor. Relative to the translated `center`, hence `Point(0,0)`.
"""
function draw(shape::CircleNode)
    return circle(Luxor.Point(0,0), shape.radius, :fill)
end

"""
Sample a random circle, return as a Matrix{Float64}.
"""
function Base.rand(rng::Random.AbstractRNG, shape::CircleNode; σ=σ_blur(shape.grid_dims))
    grid_dims = shape.grid_dims
    center = isa(shape.center, Distribution) ? rand(rng, shape.center) : shape.center
    radius = isa(shape.radius, Distribution) ? rand(rng, shape.radius) : shape.radius
    shape = CircleNode(grid_dims=grid_dims, center=center, radius=radius)
    params = [center, radius, σ]
    return (generate_shape_matrix(shape; σ=σ), params)
end

Base.rand(shape::CircleNode; kwargs...) = rand(Random.GLOBAL_RNG, shape; kwargs...)

"""
Perturb shape by adding `noise` to its parameters.
"""
perturb_sample(mainbody::CircleNode, mainbody_params, noise) = perturb_sample(Random.GLOBAL_RNG, mainbody, mainbody_params, noise)

function perturb_sample(rng::Random.AbstractRNG, mainbody::CircleNode, mainbody_params, noise)
    grid_dims = mainbody.grid_dims
    center, radius, σ = mainbody_params
    noise_scale = grid_dims[1] / 50
    𝒟_noise = Distributions.Uniform(-noise_scale*noise, noise_scale*noise)

    p_center = center .+ rand(rng, 𝒟_noise, 2)
    clamp2dims!(p_center, grid_dims)
    p_radius = clamp(radius + rand(rng, 𝒟_noise), 0.5, Inf)

    p_shape = CircleNode(grid_dims=grid_dims, center=p_center, radius=p_radius)
    p_mainbody = generate_shape_matrix(p_shape; σ=σ)
    p_mainbody_params = [p_center, p_radius, σ]

    return p_mainbody, p_mainbody_params
end

function scale_sample(mainbody::CircleNode, mainbody_params, scale)
    scale = clamp(scale, 0, 3)
    center, radius, σ = mainbody_params
    scaled_radius = sqrt(scale)*radius
    params = [center, scaled_radius, σ]
    shape = CircleNode(grid_dims=mainbody.grid_dims, center=center, radius=scaled_radius)
    return (generate_shape_matrix(shape; σ=σ), params)
end

## Elliptic Shaped Node
@with_kw struct EllipseNode <: ShapeNode
    grid_dims::Tuple{Real, Real, Real}
    center::Union{Distribution, Vector} = center_distribution(grid_dims)
    width::Union{Distribution, Real} = Distributions.Uniform(grid_dims[1]/5, grid_dims[1]/2)
    height::Union{Distribution, Real} = Distributions.Uniform(grid_dims[2]/5, grid_dims[2]/2)
    angle::Union{Distribution, Real} = Distributions.Uniform(0, 2π)
end

function draw(shape::EllipseNode)
    return ellipse(Luxor.Point(0,0), shape.width, shape.height, :fill)
end

function Base.rand(rng::Random.AbstractRNG, shape::EllipseNode; σ=σ_blur(shape.grid_dims))
    grid_dims = shape.grid_dims
    center = isa(shape.center, Distribution) ? rand(rng, shape.center) : shape.center
    width = isa(shape.width, Distribution) ? rand(rng, shape.width) : shape.width
    height = isa(shape.height, Distribution) ? rand(rng, shape.height) : shape.height
    angle = isa(shape.angle, Distribution) ? rand(rng, shape.angle) : shape.angle
    shape = EllipseNode(grid_dims=grid_dims, center=center, width=width, height=height, angle=angle)
    params = [center, width, height, angle, σ]
    return (generate_shape_matrix(shape; σ=σ), params)
end

Base.rand(shape::EllipseNode; kwargs...) = rand(Random.GLOBAL_RNG, shape; kwargs...)

perturb_sample(mainbody::EllipseNode, mainbody_params, noise) = perturb_sample(Random.GLOBAL_RNG, mainbody, mainbody_params, noise)

function perturb_sample(rng::Random.AbstractRNG, mainbody::EllipseNode, mainbody_params, noise)
    grid_dims = mainbody.grid_dims
    center, width, height, angle, σ = mainbody_params
    noise_scale = grid_dims[1] / 50
    𝒟_noise = Distributions.Uniform(-noise_scale*noise, noise_scale*noise)

    p_center = center .+ rand(rng, 𝒟_noise, 2)
    clamp2dims!(p_center, grid_dims)
    p_width = clamp(width + rand(rng, 𝒟_noise), 0.5, Inf)
    p_height = clamp(height + rand(rng, 𝒟_noise), 0.5, Inf)
    p_angle = angle + deg2rad(rand(rng, 𝒟_noise))

    p_shape = EllipseNode(grid_dims=grid_dims, center=p_center, width=p_width, height=p_height, angle=p_angle)
    p_mainbody = generate_shape_matrix(p_shape; σ=σ)
    p_mainbody_params = [p_center, p_width, p_height, p_angle, σ]

    return p_mainbody, p_mainbody_params
end

function scale_sample(mainbody::EllipseNode, mainbody_params, scale)
    scale = clamp(scale, 0, 3)
    center, width, height, angle, σ = mainbody_params
    scaled_width = sqrt(scale)*width
    scaled_height = sqrt(scale)*height
    params = [center, scaled_width, scaled_height, angle, σ]
    shape = EllipseNode(grid_dims=mainbody.grid_dims, center=center, width=scaled_width, height=scaled_height, angle=angle)
    return (generate_shape_matrix(shape; σ=σ), params)
end

## Bezier Curve Blob Shaped Node
@with_kw struct BlobNode <: ShapeNode
    grid_dims::Tuple{Real, Real, Real}
    center::Union{Distribution, Vector} = center_distribution(grid_dims)
    N::Union{Distribution, AbstractArray, Real} = Distributions.Normal(100, 1e-4)
    factor::Union{Distribution, AbstractArray, Real} = Distributions.Normal(3.5, 1e-4)
    points::Union{Nothing, Vector{Luxor.Point}} = nothing
    angle::Union{Distribution, Real} = Distributions.Uniform(0, 2π)
end

function draw(shape::BlobNode)
    return drawbezierpath(makebezierpath(shape.points), :fill)
end

function Base.rand(rng::Random.AbstractRNG, shape::BlobNode; σ=σ_blur(shape.grid_dims))
    grid_dims = shape.grid_dims
    center = isa(shape.center, Distribution) ? rand(rng, shape.center) : shape.center
    N = isa(shape.N, Distribution) || isa(shape.N, AbstractArray) ? rand(rng, shape.N) : shape.N
    factor = isa(shape.factor, Distribution) || isa(shape.factor, AbstractArray) ? rand(rng, shape.factor) : shape.factor
    if isnothing(shape.points)
        pts = polysortbyangle(randompointarray(# rng, # TODO: when Luxor.jl PR is merged.
            Luxor.Point(-grid_dims[1]/factor,-grid_dims[2]/factor),
            Luxor.Point(grid_dims[1]/factor, grid_dims[2]/factor),
            N))
    else
        pts = shape.points
    end
    angle = isa(shape.angle, Distribution) ? rand(rng, shape.angle) : shape.angle
    shape = BlobNode(grid_dims=grid_dims, center=center, N=N, factor=factor, points=pts, angle=angle)
    params = [center, N, factor, pts, angle, σ]
    return (generate_shape_matrix(shape; σ=σ), params)
end

function clamp2prior(D::Normal, x, numstd=3)
    μ = mean(D)
    span = numstd*std(D)
    x_min = μ - span
    x_max = μ + span
    return clamp(x, x_min, x_max)
end

Base.rand(shape::BlobNode; kwargs...) = rand(Random.GLOBAL_RNG, shape; kwargs...)

perturb_sample(mainbody::BlobNode, mainbody_params, noise; kwargs...) = perturb_sample(Random.GLOBAL_RNG, mainbody, mainbody_params, noise; kwargs...)

function perturb_sample(rng::Random.AbstractRNG, mainbody::BlobNode, mainbody_params, noise; recompute_points=false, copy_points=false, clamped_to_prior=true)
    grid_dims = mainbody.grid_dims
    center, N, factor, points, angle, σ = mainbody_params
    noise_scale = grid_dims[1] / 50
    𝒟_noise = Distributions.Uniform(-noise_scale*noise, noise_scale*noise)

    p_center = center .+ rand(rng, 𝒟_noise, 2)
    clamp2dims!(p_center, grid_dims)
    p_N = clamp(N + rand(rng, 𝒟_noise), 1, Inf)
    p_factor = factor + rand(rng, 𝒟_noise)
    if clamped_to_prior
        p_factor = clamp2prior(mainbody.factor, p_factor)
        p_N = clamp2prior(mainbody.N, p_N)
    end
    p_angle = angle + deg2rad(10rand(rng, 𝒟_noise))
    if recompute_points
        p_points = nothing
        p_mainbody, p_mainbody_params = rand(rng, BlobNode(grid_dims=grid_dims, center=p_center, N=p_N, factor=p_factor, points=p_points, angle=p_angle); σ=σ)
    else
        if copy_points
            p_points = deepcopy(points)
        else
            𝒟_noise_points = Distributions.Uniform(-noise_scale*noise/10, noise_scale*noise/10)
            p_points = [Luxor.Point(p.x + rand(rng, 𝒟_noise_points), p.y + rand(rng, 𝒟_noise_points)) for p in points]
        end
        p_shape = BlobNode(grid_dims=grid_dims, center=p_center, N=p_N, factor=p_factor, points=p_points, angle=p_angle)
        p_mainbody = generate_shape_matrix(p_shape; σ=σ)
        p_mainbody_params = [p_center, p_N, p_factor, p_points, p_angle, σ]
    end

    return p_mainbody, p_mainbody_params
end

function scale_sample(mainbody::BlobNode, mainbody_params, scale)
    scale = clamp(scale, 0, 3)
    center, N, factor, points, angle, σ = mainbody_params
    MineralExploration.Luxor.polyscale!(points, sqrt(scale))
    params = [center, N, factor, points, angle, σ]
    shape = BlobNode(grid_dims=mainbody.grid_dims, center=center, N=N, factor=factor, points=points, angle=angle)
    return (generate_shape_matrix(shape; σ=σ), params)
end

## Rectangular Shaped Node
@with_kw struct RectangleNode <: ShapeNode
    grid_dims::Tuple{Real, Real, Real}
    center::Union{Distribution, Vector} = center_distribution(grid_dims)
    width::Union{Distribution, Real} = Distributions.Uniform(grid_dims[1]/3.5, grid_dims[1]/2.5)
    height::Union{Distribution, Real} = Distributions.Uniform(grid_dims[2]/3.5, grid_dims[2]/2.5)
    angle::Union{Distribution, Real} = Distributions.Uniform(0, 2π)
end

function draw(shape::RectangleNode)
    cornerpoint = Luxor.Point(-shape.width/2, -shape.height/2)
    return rect(cornerpoint, shape.width, shape.height, :fill)
end

function Base.rand(rng::Random.AbstractRNG, shape::RectangleNode; σ=σ_blur(shape.grid_dims))
    grid_dims = shape.grid_dims
    center = isa(shape.center, Distribution) ? rand(rng, shape.center) : shape.center
    width = isa(shape.width, Distribution) ? rand(rng, shape.width) : shape.width
    height = isa(shape.height, Distribution) ? rand(rng, shape.height) : shape.height
    angle = isa(shape.angle, Distribution) ? rand(rng, shape.angle) : shape.angle
    shape = RectangleNode(grid_dims=grid_dims, center=center, width=width, height=height, angle=angle)
    params = [center, width, height, angle, σ]
    return (generate_shape_matrix(shape; σ=σ), params)
end

Base.rand(shape::RectangleNode; kwargs...) = rand(Random.GLOBAL_RNG, shape; kwargs...)

perturb_sample(mainbody::RectangleNode, mainbody_params, noise) = perturb_sample(Random.GLOBAL_RNG, mainbody, mainbody_params, noise)

function perturb_sample(rng::Random.AbstractRNG, mainbody::RectangleNode, mainbody_params, noise)
    grid_dims = mainbody.grid_dims
    center, width, height, angle, σ = mainbody_params
    noise_scale = grid_dims[1] / 50
    𝒟_noise = Distributions.Uniform(-noise_scale*noise, noise_scale*noise)

    p_center = center .+ rand(rng, 𝒟_noise, 2)
    clamp2dims!(p_center, grid_dims)
    p_width = clamp(width + rand(rng, 𝒟_noise), 0.5, Inf)
    p_height = clamp(height + rand(rng, 𝒟_noise), 0.5, Inf)
    p_angle = angle + deg2rad(rand(rng, 𝒟_noise))

    p_shape = RectangleNode(grid_dims=grid_dims, center=p_center, width=p_width, height=p_height, angle=p_angle)
    p_mainbody = generate_shape_matrix(p_shape; σ=σ)
    p_mainbody_params = [p_center, p_width, p_height, p_angle, σ]

    return p_mainbody, p_mainbody_params
end

function scale_sample(mainbody::RectangleNode, mainbody_params, scale)
    scale = clamp(scale, 0, 3)
    center, width, height, angle, σ = mainbody_params
    scaled_width = sqrt(scale)*width
    scaled_height = sqrt(scale)*height
    params = [center, scaled_width, scaled_height, angle, σ]
    shape = RectangleNode(grid_dims=mainbody.grid_dims, center=center, width=scaled_width, height=scaled_height, angle=angle)
    return (generate_shape_matrix(shape; σ=σ), params)
end

## Multi-Shape Node
struct MultiShapeNode <: MainbodyGen
    shapes::Vector{ShapeNode}
end

function Base.rand(rng::Random.AbstractRNG, shape::MultiShapeNode; σ=[σ_blur(s.grid_dims) for s in shape.shapes])
    shapes = []
    params = []
    for (i,s) in enumerate(shape.shapes)
        sampled_shape, sampled_params = rand(rng, s; σ=σ[i])
        push!(shapes, sampled_shape)
        push!(params, sampled_params)
    end
    return sum(shapes), params
end

Base.rand(shape::MultiShapeNode; kwargs...) = rand(Random.GLOBAL_RNG, shape; kwargs...)

perturb_sample(mainbody::MultiShapeNode, mainbody_params, noise) = perturb_sample(Random.GLOBAL_RNG, mainbody, mainbody_params, noise)

function perturb_sample(rng::Random.AbstractRNG, mainbody::MultiShapeNode, mainbody_params, noise)
    p_mainbody = []
    p_mainbody_params = []
    for (i,s) in enumerate(mainbody.shapes)
        p_mainbody_s, p_mainbody_params_s = perturb_sample(rng, s, mainbody_params[i], noise)
        push!(p_mainbody, p_mainbody_s)
        push!(p_mainbody_params, p_mainbody_params_s)
    end

    return sum(p_mainbody), p_mainbody_params
end

function scale_sample(mainbody::MultiShapeNode, mainbody_params, scale)
    error("`scale_sample` for MultiShapeNode is not implemented.")
end
