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
%%% load all tests
% u = {};
% v = {};
% phi = {};
% count = 1;
% for i = 1:9
%     for j = 1:6
%         name = strcat("test_", num2str(i), "/dFacto", num2str(j), "/dFacto", num2str(j), ".mat" );
%         data = load(name);
%         u{count} = data.u;
%         v{count} = data.v;
%         phi{count} = data.phi;
%         count = count + 1;
%     end
% end

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

intv = 10;
for i = 1:9
    phi{i} = phi{i}(1:intv:end, :);
    u{i} = u{i}(1:intv:end, :);
    v{i} = v{i}(1:intv:end, :);
end
phi_init = phi_init(1:intv:end, :);
x = x(1:intv:end, :);
y = y(1:intv:end, :);

int_step = 3;
for i = 1:9
    phi{i} = phi{i}(:, 1:int_step:end);
    u{i} = u{i}(:, 1:int_step:end);
    v{i} = v{i}(:, 1:int_step:end);
end
force_df1 = force_df1(1:int_step:end);
% scatter(x, y, [], phi_init(:, 1))

%%
num_1 = ceil(11236/intv);
num_f = ceil(23/int_step);

phi_init_train = zeros(num_1*9*num_f, num_1);
for i = 1:9
    for j = 1:num_1*num_f
        phi_init_train( (i-1)*(num_1*num_f)+j, :) = phi_init(:, i)';
    end
end

for i = 1:10
scatter(x, y, [], phi_init_train(i, :))
end

for i = 1:num_f*9
    x_train((i-1)*num_1+1:i*num_1, 1) = x;
    x_train((i-1)*num_1+1:i*num_1, 2) = y;
end
for i = 1:num_1*9
    stress_train((i-1)*num_f+1: i*num_f, 1) = force_df1;
end
count = 0;
for i = 1:9
    for j = 1:num_f
        y_train(count+1:count+num_1, 1) = phi{i}(:, j);
        count = count + 450;
    end
end

clear name file_name count i j data intv int_step force_df1 num num_1 num_f phi u v x y phi_init

save dataset.mat






