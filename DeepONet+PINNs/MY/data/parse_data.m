clear all
close all
clc

num_case = 100;
dim = 10000;
dim_f = 100;
X_data = zeros(dim*num_case, 2);
U_data = zeros(dim*num_case, 1);
F_data = zeros(dim*num_case, dim_f);
F_PDE_data = zeros(dim*num_case, 1);

for id = 1:num_case
    file_name = strcat("d/d_", num2str(id-1), ".dat");
    d_dataset = load(file_name);
    X_data((id-1)*dim+1:id*dim, :) = d_dataset(:, 1:2);
    U_data((id-1)*dim+1:id*dim, :) = d_dataset(:, 3);
    
    file_name = strcat("s/s_", num2str(id-1), ".dat");
    func = load(file_name);
    
    s_dataset = repmat(transpose(func), [dim, 1]);
    F_data((id-1)*dim+1:id*dim, :) = s_dataset;
    
    f_dataset = repmat(func, [dim_f, 1]);
    
    F_PDE_data((id-1)*dim+1:id*dim, :) = f_dataset;
end

clear d_dataset file_name func id s_dataset f_dataset
save dataset.mat