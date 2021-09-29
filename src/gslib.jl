@with_kw struct GSLIBDistribution
    grid_dims::Tuple{Int64, Int64, Int64} = (80, 80, 1)
    n::Tuple{Int64, Int64, Int64} = (80, 80, 1) # same as grid_dims, renamed for convenience
    data::RockObservations = RockObservations()
    nugget::Tuple{Int64, Int64} = (1, 0) # TODO Check with Anthony if Int required
    variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0) # TODO Check with Anthony if Int required
    # CHANGE RARELY
    target_histogram_file::String = "parameters/example_porosity.txt"
    columns_for_vr_and_wt = (1,0)
    zmin_zmax = (0.1, 0.6)
    lower_tail_option = (1, 0.1)
    upper_tail_option = (1, 0.6)

    # DO NOT CHANGE BELOW PARAMS
    transform_data::Bool = true
    mn = (0.5, 0.5, 0.5)
    sz = (1,1,1)
end

function write_string_to_file(fn, str)
    open(fn, "w") do io
        write(io, str)
    end
    fn
end

function data_to_string(data::RockObservations)
    str = """
    data
    4
    x
    y
    z
    poro
    """
    #TODO: Deal with cell-centered offsets?
    for i=1:length(data)
        str = string(str, data.coordinates[1,i] - 1, " ", data.coordinates[2,i] - 1, " 0.5 ", data.ore_quals[i], "\n")
    end
    str
end

function params_to_string(p::GSLIBDistribution, data_file, N, dir, seed=nothing)
    if seed == nothing
        seed = rand(1:10000000)
    end
    """
    Parameters for SGSIM
********************

START OF PARAMETERS:
$(data_file)          -file with data
1  2  3  4  0  0              -  columns for X,Y,Z,vr,wt,sec.var.
-9999999 999999               -  trimming limits
$(Int(p.transform_data))                             -transform the data (0=no, 1=yes)
$(dir)sgsim.trn                     -  file for output trans table
1                             -  consider ref. dist (0=no, 1=yes)
$(p.target_histogram_file)                  -  file with ref. dist distribution
$(p.columns_for_vr_and_wt[1])  $(p.columns_for_vr_and_wt[2])                          -  columns for vr and wt
$(p.zmin_zmax[1])    $(p.zmin_zmax[2])                     -  zmin,zmax(tail extrapolation)
$(p.lower_tail_option[1])       $(p.lower_tail_option[2])                     -  lower tail option, parameter
$(p.upper_tail_option[1])      $(p.upper_tail_option[2])                     -  upper tail option, parameter
1                             -debugging level: 0,1,2,3
$(dir)sgsim.dbg                     -file for debugging output
$(dir)sgsim.out                     -file for simulation output
$N                             -number of realizations to generate
$(p.n[1])    $(p.mn[1])    $(p.sz[1])              -nx,xmn,xsiz
$(p.n[2])    $(p.mn[2])    $(p.sz[1])              -ny,ymn,ysiz
$(p.n[3])    $(p.mn[3])    $(p.sz[1])              -nz,zmn,zsiz
$seed                         -random number seed
0     8                       -min and max original data for sim
12                            -number of simulated nodes to use
1                             -assign data to nodes(0=no, 1=yes)
1     3                       -multiple grid search(0=no,
0                             -maximum data per octant (0=not
100.0  100.0  10.0              -maximum search radii
0.0   0.0   0.0              -angles for search ellipsoid
51    51    11                -size of covariance lookup table
0     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,
../data/ydata.dat             -  file with LVM, EXDR, or COLC
4                             -  column for secondary variable
$(p.nugget[1])    $(p.nugget[2])                      -nst, nugget effect
$(p.variogram[1])    $(p.variogram[2])  $(p.variogram[3])   $(p.variogram[4])   $(p.variogram[5])     -it,cc,ang1,ang2,ang3
$(p.variogram[6])  $(p.variogram[7])  $(p.variogram[8])     -a_hmax, a_hmin, a_vert
"""
end

function write_params_to_file(p::GSLIBDistribution, N; dir="./", out_fn="sgsim.par", data_fn="data.txt")
    data_file = write_string_to_file(string(dir, data_fn), data_to_string(p.data))
    write_string_to_file(string(dir, out_fn), params_to_string(p, data_file, N, dir))
end

# run_quiet =

function Base.rand(p::GSLIBDistribution, n::Int64=1, dir="sgsim_output/"; silent::Bool=true)
    # Write the param file
    if silent
        stdout_orig = stdout
        redirect_stdout()
    end
    fn = write_params_to_file(p, n; dir=dir) # NOTE: If we are going to want to sample many instances then we can include an "N" parameter here instead of the 1, but would need to update the code below as well

    # Run sgsim
    run(`sgsim $fn`)

    # Load the results and return
    vals = CSV.File("$(dir)sgsim.out",header=3) |> CSV.Tables.matrix
    # reshape(vals, p.n..., N) # For multiple samples

    poro_2D = reshape(vals, p.n)
    ore_quals = repeat(poro_2D, outer=(1, 1, 8))
    if silent
        redirect_stdout(stdout_orig)
    end
    return ore_quals
end

Base.rand(rng::Random.AbstractRNG, p::GSLIBDistribution, n::Int64=1, dir::String="sgsim_output/") = Base.rand(p, n, dir)
