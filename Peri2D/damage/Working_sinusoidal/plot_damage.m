
% This script is used to check the sharpness of the damage field
clc
clear

dataset = load('dataset.mat');
x= dataset.x_train;
y = dataset.y_train;
damage = dataset.damage(:,1);
scatter(x,y,[],damage)
xyd = [x y damage];

for i = 1:97
    
    figure;
    temp = xyd((i-1)*19+1:(i*19),:);
    plot(temp(:,2),temp(:,3))
    temp(:, 3)
    name = strcat('x_',num2str(i),'.png');
    saveas(gcf,name)
    
end

close all