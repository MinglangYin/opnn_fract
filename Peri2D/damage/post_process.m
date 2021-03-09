clear
clc

locationFolder = pwd;
filePattern = fullfile(locationFolder,'*.txt');% Get a directory listing
textFiles = dir(filePattern);
sizeA = [2 100];
formatSpec = '%f %f';

filename = sprintf('0.txt');
fileID = fopen(filename,'r');   
textFile = fscanf(fileID,formatSpec,sizeA);
% x_train = 0.01*textFile(1,:)';
% y_train = 0.2*ones(size(x_train));

for ifile = 0 : length(textFiles)-1
    
    filename = sprintf('%d.txt',ifile);
    fileID = fopen(filename,'r');   
    textFile = fscanf(fileID,formatSpec,sizeA)';
    boundary_train(:,ifile+1) = textFile(1:2:end,2);
end

save('dataset.mat','boundary_train')