abstract type MainbodyGen end

@with_kw struct SingleFixedNode <: MainbodyGen
    mainbody_weight::Float64 = 0.45
    mainbody_loc::Vector{Float64} = [25.0, 25.0]
    mainbody_var_min::Float64 = 40.0
    mainbody_var_max::Float64 = 80.0
end
