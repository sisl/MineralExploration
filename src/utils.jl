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
    return plot(μ, yerror=σ)
end
