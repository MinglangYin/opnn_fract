clear
clc

locationFolder = pwd;
filePattern_old = fullfile(locationFolder,'Data_old/force_center/*.txt');% Get a directory listing
textFiles_old = dir(filePattern_old);
filePattern_new = fullfile(locationFolder,'Data_new/force_center/*.txt');% Get a directory listing
textFiles_new = dir(filePattern_new);
sizeA = [2 100];
formatSpec = '%f %f';
cd Data_old/force_center

% filename = sprintf('0.txt');
% fileID = fopen(filename,'r');
% textFile = fscanf(fileID,formatSpec,sizeA);
% x_train = 0.01*textFile(1,:)';
% y_train = 0.2*ones(size(x_train));

for ifile = 0 : length(textFiles_old)-1
    
    filename = sprintf('%d.txt',ifile);
    fileID = fopen(filename,'r');
    textFile = fscanf(fileID,formatSpec,sizeA)';
    boundary_train(:,ifile+1) = textFile(1:2:end,2);
end

cd ../
cd test_BC_center
cd test_0

filename = sprintf('final_terror.csv');
trainTable = readtable(filename);
trainData = table2array(trainTable);
x_train = trainData(:,2);
y_train = trainData(:,3);
[index_y] = find((y_train >= 0) & (y_train < 1.85) & (x_train > 0.2) & (x_train < 9.85));
y_train = y_train(index_y,1);
x_train = x_train(index_y,1);

numData = length(x_train);
total_cases = length(textFiles_old)+length(textFiles_new);
damage = zeros(numData, total_cases);
cd ../

for iFolder = 0 : length(textFiles_old)-1
    
    folderName = sprintf('test_%d',iFolder);
    cd(folderName)
    filename = sprintf('final_terror.csv');
    trainTable = readtable(filename);
    trainData = table2array(trainTable);
    damage(:,iFolder+1) = trainData(index_y,4);
    cd ../
end
cd ../../
cd Data_new/force_center

for ifile = 0 : length(textFiles_new)-1
    
    filename = sprintf('%d.txt',ifile);
    fileID = fopen(filename,'r');
    textFile = fscanf(fileID,formatSpec,sizeA)';
    boundary_train(:,100+ifile+1) = textFile(1:2:end,2);
end

cd ../
cd test_BC_center

for ifile = 0 : length(textFiles_new)-1
    
    filename = sprintf('final_terror_%d.csv',ifile);
    trainTable = readtable(filename);
    trainData = table2array(trainTable);
    damage(:,100+ifile+1) = trainData(index_y,4);
    
end

cd ../../../
x_train = reshape(x_train,74,386);
y_train = reshape(y_train,74,386);
x_train = x_train(:,1:2:end);
y_train = y_train(:,1:2:end);
x_train = reshape(x_train,74*193,1);
y_train = reshape(y_train,74*193,1);

damage1 = zeros(74*193,total_cases);

for idamage = 1:total_cases
    temp = damage(:,idamage);
    temp = reshape(temp,74,386);
    temp = temp(:,1:2:end);
    damage1(:,idamage) = reshape(temp,74*193,1);
    damage1(:,idamage) = imgaussfilt(damage1(:,idamage),1e-8);
end

damage = damage1;
clear damage1
damage = normalize(damage,'range');

% damage(damage > 1e-2) = 1.0;
% damage(damage < 1) = 0.0;
% scatter(x_train,y_train,[],damage(:,1))
% 
% damage1 = imgaussfilt(damage(:,1),1e-8);
scatter(x_train,y_train,[],damage(:,1));

save('dataset.mat','boundary_train','x_train','y_train','damage')


