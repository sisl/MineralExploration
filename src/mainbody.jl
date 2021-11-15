abstract type MainbodyGen end

## Single Fixed Node
@with_kw struct SingleFixedNode <: MainbodyGen
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    mainbody_loc::Vector{Float64} = [25.0, 25.0]
    mainbody_var_min::Float64 = 40.0
    mainbody_var_max::Float64 = 80.0
end

function Base.rand(rng::Random.AbstractRNG, mb::SingleFixedNode)
    x_dim = mb.grid_dims[1]
    y_dim = mb.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    mainbody_var = rand(rng)*(mb.mainbody_var_max - mb.mainbody_var_min) + mb.mainbody_var_min
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mb.mainbody_loc, cov)
    for i = 1:x_dim
        for j = 1:y_dim
            lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    return (lode_map, mainbody_var)
end

Base.rand(mb::SingleFixedNode) = rand(Random.GLOBAL_RNG, mb)

function perturb_sample(mb::SingleFixedNode, mainbody_var::Float64, noise::Float64)
    mainbody_var += 2.0*(rand() - 0.5)*noise
    mainbody_var = clamp(mainbody_var, 0.0, Inf)
    mainbody_map = zeros(Float64, Int(mb.grid_dims[1]), Int(mb.grid_dims[2]))
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mb.mainbody_loc, cov)
    for i = 1:mb.grid_dims[1]
        for j = 1:mb.grid_dims[2]
            mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    return (mainbody_map, mainbody_var)
end

## Single Variable Node
@with_kw struct SingleVarNode <: MainbodyGen
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    mainbody_loc_bounds::Vector{Float64} = [20.0, 30.0]
    mainbody_var_min::Float64 = 40.0
    mainbody_var_max::Float64 = 80.0
end

function Base.rand(rng::Random.AbstractRNG, mb::SingleVarNode) #TODO
    x_dim = mb.grid_dims[1]
    y_dim = mb.grid_dims[2]
    lode_map = zeros(Float64, x_dim, y_dim)
    mainbody_var = rand(rng)*(mb.mainbody_var_max - mb.mainbody_var_min) + mb.mainbody_var_min
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mb.mainbody_loc, cov)
    for i = 1:x_dim
        for j = 1:y_dim
            lode_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    return (lode_map, mainbody_var)
end

Base.rand(mb::SingleFixedNode) = rand(Random.GLOBAL_RNG, mb)

function perturb_sample(mb::SingleVarNode, mainbody_var::Float64, noise::Float64) # TODO
    mainbody_var += 2.0*(rand() - 0.5)*noise
    mainbody_var = clamp(mainbody_var, 0.0, Inf)
    mainbody_map = zeros(Float64, Int(mb.grid_dims[1]), Int(mb.grid_dims[2]))
    cov = Distributions.PDiagMat([mainbody_var, mainbody_var])
    mvnorm = MvNormal(mb.mainbody_loc, cov)
    for i = 1:mb.grid_dims[1]
        for j = 1:mb.grid_dims[2]
            mainbody_map[i, j] = pdf(mvnorm, [float(i), float(j)])
        end
    end
    return (mainbody_map, mainbody_var)
end
