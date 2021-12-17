using CSV

schedule = CSV.read("./data/tests/random/schedule.csv", CSV.Tables.matrix, header=1)

n = size(schedule)[1]
for i=1:n
    ARGS = schedule[i, :]
    include("./scripts/run_random_test.jl")
end
