
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
    count_list = [length(ele) for ele in a_list]
    combination_list = [push!(combination_list,1:i) for i in count_list][1]
    total_row = prod(count_list)
    reference_table = reshape(collect(Iterators.product(combination_list...)),total_row,1)
    return reference_table
end


function get_hyperParameters(index_from_script)
    K_choices = [1, 2, 4]
    alpha_choices = [0.0, 1.1, 2.2]
    UCB_choices = [1.2, 1.3, 1.4]
    leaf_est_choices= ["fn", 0.0]
    varible_options = [K_choices,alpha_choices,UCB_choices,leaf_est_choices]

    reference = look_up_table(varible_options)
    reference_index = reference[index_from_script]
    return_values = []
    for i in 1:length(reference_index)
        push!(return_values,varible_options[i][reference_index[i]])

    end


    return return_values
end