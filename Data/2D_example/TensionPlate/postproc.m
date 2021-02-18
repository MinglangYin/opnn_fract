clear all
close all
clc

file_name = "Supplementary/cordinates.mat"; 
data = load(file_name);
x = data.cordinates(:, 1);
y = data.cordinates(:, 2);

file_name = ["initPhi", "initPhi_lBottom", "initPhi_lBottom_long", "initPhi_lBottom_short", ...
    "initPhi_long", "initPhi_short", "initPhi_uBottom", "initPhi_uBottom_long", ...
    "initPhi_uBottom_short"];

count = 1;
for name = file_name
    file_name = strcat("Supplementary/", name, ".mat"); 
    data = load(file_name);
    phi_init(:, count) = data.Phi;
    count = count + 1;
end

% for i =1:9
%     scatter(x, y, [], phi_init(:, i))
% end

force_df1 = load("Supplementary/loading_f_df1.txt");

%%
u = {};
v = {};
phi = {};
count = 1;
for i = 1:9
    name = strcat("test_", num2str(i), "/dFacto1/dFacto1.mat" );
    data = load(name);
    u{count} = data.u;
    v{count} = data.v;
    phi{count} = data.phi;
    count = count + 1;
end

%%
intv = 3;
num = ceil(23/intv);
phi_data = {};
u_data = {};
v_data = {};
for i = 1:9
    phi_data{i} = phi{i}(:, 1:intv:end);
    u_data{i} = u{i}(:, 1:intv:end);
    v_data{i} = v{i}(:, 1:intv:end);
end

%%
x_train = linspace(0, 1, 110);
y_train = linspace(0, 1, 110);
stress_train = force_df1(1:intv:end);

count = 1;
for i = 1:9
    for j = 1:num
        phi_train(:, count) = phi_data{i}(:, j);
        phi_init_train(:, count) = phi_init(:, i);
        u_train(:, count) = u_data{i}(:, j);
        v_train(:, count) = v_data{i}(:, j);
        count = count + 1;
    end
end

clear name file_name count i j data intv int_step force_df1 
clear num num_1 num_f phi u v x y phi_init phi_data
clear u_data v_data

save dataset.mat






