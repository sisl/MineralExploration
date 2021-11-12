struct GPNextAction
    l::Float64
    sf::Float64
    sn::Float64
end

gp_action = GPNextAction(1.0, 15.0, 15.0)

function kernel(x::Tuple{Bool, CartesianIndex{2}}, y::Tuple{Bool, CartesianIndex{2}}, l::Float64, sf::Float64)
    d = (x[2][1] - y[2][1])^2 + (x[2][2] - y[2][2])^2
    return sf*exp(-d/l^2)
end

function gp_posterior(X::Vector{Tuple{Bool, CartesianIndex{2}}},
                    x::Vector{Tuple{Bool, CartesianIndex{2}}}, y::Vector{Float64},
                    ns, l, sf, sn)
    m = length(X)
    n = length(x)
    Kxx = zeros(Float64, n, n)
    Kxz = zeros(Float64, n, m)
    Kzz = zeros(Float64, m)
    for i = 1:n
        for j = 1:n
            Kxx[i, j] = kernel(x[i], x[j], l, sf)
            if i == j
                Kxx[i, j] += sn/ns[i]
            end
        end
        for j = 1:m
            Kxz[i, j] = kernel(x[i], X[j], l, sf)
            Kzz[j] = kernel(X[j], X[j], l, sf)
        end
    end
    α = inv(Kxx)
    σ² = Kzz - diag(transpose(Kxz)*α*Kxz, 0)
    μ = transpose(Kxz)*α*y
    return (μ, σ²)
end

function approx_posterior(X::Vector{Tuple{Bool, CartesianIndex{2}}},
                    x::Vector{Tuple{Bool, CartesianIndex{2}}}, y::Vector{Float64},
                    ns, l, sf, sn)
    m = length(X)
    n = length(x)
    W = zeros(Float64, m, n)
    for i = 1:m
        for j = 1:n
            W[i, j] = kernel(X[i], x[j], l, sf)/sf*ns[j]
        end
    end

    w = sum(W, dims=2)
    σ² = sf./(1.0 .+ w)
    μ = W*y./w
    return (μ, σ²)
end

function expected_improvement(μ, σ², f)
    σ = sqrt.(σ²)
    dist = Normal(0.0, 1.0)
    Δ = μ .- f
    δ = Δ./σ
    ei = Δ.*(Δ .>= 0.0)
    for (i, d) in enumerate(δ)
        ei[i] += σ[i]*pdf(dist, d) - abs(Δ[i])*cdf(dist, d)
    end
    return ei
end

function POMCPOW.next_action(o::GPNextAction, pomdp::GaussianPOMDP, ::Any, h::POMCPOW.BeliefNode)
    a_idxs = h.tree.tried[h.node]
    tried_actions = h.tree.a_labels[a_idxs]::Vector{Tuple{Bool, CartesianIndex{2}}}
    action_values = h.tree.v[a_idxs]
    action_ns = h.tree.n[a_idxs]
    actions = POMDPs.actions(pomdp)::Vector{Tuple{Bool, CartesianIndex{2}}}

    if length(tried_actions) > 0
        # μ, σ² = gp_posterior(actions, tried_actions, action_values, action_ns, o.l, o.sf, o.sn)
        μ, σ² = approx_posterior(actions, tried_actions, action_values, action_ns, o.l, o.sf, o.sn)
        f = maximum(action_values)
        ei = expected_improvement(μ, σ², f)
        a_idx = argmax(ei)
        a = actions[a_idx]
    else
        a = rand(actions)
    end
    return a
end
