# = MRST_julia.jl

function initializeMATLABSim(nl::Real, nw::Real, nd::Real,
                            # l::Real, w::Real, d::Real,
                            # interval::Real, total_time::Real,
                            MRST_dir::String, print_matlab::Bool)
    println("Initializing MATLAB session...")
    if print_matlab
        session = MSession()
    else
        session = MSession(0)
    end 
    eval_string(session, "addpath('$MRST_dir')")
    eval_string(session, "addpath('matlab')")
    eval_string(session, "startup")
    eval_string(session, "clear")
    grid_array = Float64[nl nw nd]
    put_variable(session, :GRID_ARRAY, grid_array)

    t_total = 530.0
    t_injection = 30.0
    n_injection_step = 30.0
    n_shutin_step = 20.0
    interval = 5.0

    put_variable(session, :TOTAL_TIME, t_total) # in years
    put_variable(session, :INJECTION_STOP, t_injection) # in years from beginning
    put_variable(session, :NSTEP_INJECTION, n_injection_step) # number of MRST simulation steps/numerical
    put_variable(session, :NSTEP_SHUT_IN, n_shutin_step)# number of MRST simulation steps
    put_variable(session, :INTERVAL, interval) # time between wells
    # domain_array = Float64[l w d]
    # put_variable(session, :DOMAIN_ARRAY, domain_array)
    # put_variable(session, :INTERVAL, Float64[interval])
    # put_variable(session, :TOTAL_TIME,  Float64[total_time])
    eval_string(session, "initialize2DReservoirModel") # Entry Point
    println("MATLAB Session Opened")
    return session
end

function runMATLABSim(session, porosity::Array{Float64, 3},
                    permeability::Array{Float64, 3}, w_coords::Array
                    # c_rock::Real, srw::Real, src::Real, pe::Real, muw::Real,
                    # t_total::Real, t_injection::Real, n_injection_step::Real,
                    # n_shutin_step::Real, interval::Real
                    )
    c_rock  = 4.35e-5
    srw     = 0.27
    src     = 0.20
    pe      = 5.0
    muw     = 8e-4

    paramters = Float64[c_rock srw src pe muw]
    put_variable(session, :PARAMETERS, paramters)
    put_variable(session, :POROSITY, porosity)
    put_variable(session, :PERMEABILITY, permeability)

    put_variable(session, :W_COORDS, w_coords)

    eval_string(session, "run2DReservoirSim") # Entry Point

    injector_bhps = jarray(get_mvariable(session, :injector_bhps))
    mass_fractions = jarray(get_mvariable(session, :mass_fractions))
    time_days = jarray(get_mvariable(session, :time_days))
    pressure_map_first_layer = jarray(get_mvariable(session, :pressure_map_first_layer))
    observation_sat = jarray(get_mvariable(session, :observation_sat))
    schedule_idx = jarray(get_mvariable(session, :schedule_idx))
    # # TODO: Remove (debugging only)
    # cumtime = jarray(get_mvariable(session, :cumtime))
    # println(cumtime)
    return (schedule_idx, injector_bhps, mass_fractions, time_days, pressure_map_first_layer, observation_sat)
end

function MRST_reward(schedule_idx, injector_bhps, mass_fractions_padded, time_days, pressure_map_first_layer, observation_sat)
            # % legend names
            # names = {'Dissolved'           , 'Structural residual' , ...
            #          'Residual'            , 'Residual in plume'  , ...
            #          'Structural subscale' , 'Structural plume'   , ...
            #          'Free plume'          , 'Exited'};


    schedule_idx_v = Int.(vec(schedule_idx))

    schedule_index_unique = unique(schedule_idx_v)
    search_idx = []

    mass_fractions = mass_fractions_padded[2:end,:] # take the added 0 out

    reward=[]

    for i in 1:length(schedule_index_unique)


        current_idx = schedule_index_unique[i]
        push!(search_idx, current_idx)

        current_indices=indexin(schedule_idx_v, search_idx)


        injector_bhps_temp = Array{Float64, 2}(undef, sum(schedule_idx_v.==current_indices), size(injector_bhps)[2])
        [injector_bhps_temp[:,i] = injector_bhps[:,i][schedule_idx_v.==current_indices] for i in 1:size(injector_bhps)[2]]


        mass_fractions_temp = Array{Float64, 2}(undef, sum(schedule_idx_v.==current_indices), size(mass_fractions)[2])
        [mass_fractions_temp[:,i] = mass_fractions[:,i][schedule_idx_v.==current_indices] for i in 1:size(mass_fractions)[2]]


        pressure_map_first_layer_temp = Array{Float64, 2}(undef, sum(schedule_idx_v.==current_indices), size(pressure_map_first_layer)[2])
        [pressure_map_first_layer_temp[:,i] = pressure_map_first_layer[:,i][schedule_idx_v.==current_indices] for i in 1:size(pressure_map_first_layer)[2]]

        observation_sat_temp = observation_sat[schedule_idx_v.==current_indices]


        #mass_fract_reward = mass_fractions[2:end,:]
        existed_vol = mass_fractions_temp[:,8][end]/mean(mass_fractions_temp[:,8])
        free_plume_vol = mass_fractions_temp[:,7][end]/mean(mass_fractions_temp[:,7])
        trapped_vol = sum(mass_fractions_temp[:,1:6],dims=2)[end]/mean(sum(mass_fractions_temp[:,1:6],dims=2))
        pressure = max(maximum(injector_bhps_temp),maximum(pressure_map_first_layer_temp))
        initial_pressure =  maximum(pressure_map_first_layer_temp[1,:]) #TODO check
        reward_temp =(pressure>initial_pressure*1.2)*(-1000) + existed_vol*(-1000) + free_plume_vol *(-5) + trapped_vol*(10)

        push!(reward, reward_temp)



    end

    return reward


end
