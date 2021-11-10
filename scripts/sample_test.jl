
using Combinatorics
Combinatorics
K = [1, 2, 4]
alpha = [0.0, 1.1, 2.2]
UCB = [1.2, 1.3, 1.4]
leaf_est= [0, 1]

x = combinations([[1,2,3],[1,2,3]])

for c in x
    println(c)
end



function look_up_table(a_list)

    count_list = []
    combination_list = []
    # for i in a_list:
    #     push!(count_list,length(i))
    # end
    count_list = [length(ele) for ele in a_list]
    combination_list = [push!(combination_list,1:i) for i in count_list][1]
    total_row = prod(count_list)
    reference_table = reshape(collect(Iterators.product(combination_list...)),total_row,1)
    return reference_table
end
