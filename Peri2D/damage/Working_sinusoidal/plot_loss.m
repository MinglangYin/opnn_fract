clc
clear
epochs = 10000;

sizeA = [1 epochs];
formatSpec = '%f';
loss = zeros(epochs,2);
filename = sprintf('train_loss.txt');
fileID = fopen(filename,'r');
textFile = fscanf(fileID,formatSpec,sizeA)';
loss(:,1) = 1:epochs;
loss(:,2) = textFile(:,1);

semilogy(loss(:,1),loss(:,2));
hold on

filename = sprintf('test_loss.txt');
fileID = fopen(filename,'r');
textFile = fscanf(fileID,formatSpec,sizeA)';
loss(:,1) = 1:epochs;
loss(:,2) = textFile(:,1);
semilogy(loss(:,1),loss(:,2));

legend('Train Loss','Test Loss')
saveas(gcf, 'loss.fig')