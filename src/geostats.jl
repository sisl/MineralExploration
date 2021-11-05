@with_kw struct GeoStatsDistribution # Only a distribution over the rock properties right now
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    data::RockObservations = RockObservations()
    domain::CartesianGrid{2, Int64} = CartesianGrid{Int64}(50, 50)
    mean::Float64 = 0.3
    variogram::Variogram = SphericalVariogram(sill=0.005, range=30.0,
                                            nugget=0.0001)
end

function Base.rand(rng::AbstractRNG, d::GeoStatsDistribution, n::Int64=1)
    if isempty(d.data.coordinates) # Unconditional simulation
        # println("HERE")
        problem = SimulationProblem(d.domain, (:ore => Float64), n)
        solver = LUGS(
                            :ore => (
                                        mean=0.0,
                                        variogram=d.variogram
                                           )
                             )
    else # Conditional simulation
        table = DataFrame(ore=d.data.ore_quals .- d.mean)
        domain = PointSet(d.data.coordinates)
        geodata = georef(table, domain)
        problem = SimulationProblem(geodata, d.domain, (:ore), n)
        solver = LUGS(
                            :ore => (
                                        variogram=d.variogram,
                                           )
                             )
    end
    # solver = SGS(
    #                     :ore => ( mean=d.mean,
    #                                 variogram=d.variogram,
    #                                 neighborhood=NormBall(100.0),
    #                                 maxneighbors=10,
    #                                 path=RandomPath()
    #                                    )
    #                      )
     # solver = FFTGS(
     #                     :ore => ( mean=d.mean,
     #                                 variogram=d.variogram
     #                                    )
     #                      )
    solution = GeoStats.solve(problem, solver)
    ore_maps = Array{Float64, 3}[]
    for s in solution[:ore]
        ore_2D = reshape(s, d.grid_dims) .+ d.mean
        ore_map = repeat(ore_2D, outer=(1, 1, 8))
        push!(ore_maps, ore_map)
    end
    if n == 1
        return ore_maps[1]
    else
        return ore_maps
    end
end

Base.rand(d::GeoStatsDistribution, n::Int64=1) = Base.rand(Random.GLOBAL_RNG, d, n)

function solve_gp(d::GeoStatsDistribution)
    table = DataFrame(porosity=d.data.ore_quals)
    domain = PointSet(d.data.coordinates)
    geodata = georef(table, domain)
    problem = EstimationProblem(geodata, d.domain, :ore)
    solver = Kriging(
                        :ore => ( mean=d.mean,
                                    variogram=d.variogram
                                       )
                         )
    solution = GeoStats.solve(problem, solver)
    ore_mean = reshape(solution[:ore], d.grid_dims)
    ore_var = reshape(solution[:ore_variance], d.grid_dims)
    return (ore_mean, ore_var)
end
