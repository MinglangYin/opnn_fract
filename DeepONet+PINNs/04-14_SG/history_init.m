% clear 
% clc

crackTipLoc = [0.5, 0.5; 0.2, 0.5; 0.2, 0.65;
            0.2, 0.3; 0.5, 0.65; 0.5, 0.3;
            0.35, 0.5; 0.35, 0.65; 0.35, 0.3];
l = 0.0125;
constB = 1;
G = 2.7;
data = load("dataset_prep.mat");
u_train = data.u_train;
v_train = data.v_train;
phi_train = data.phi_train;
stress_train = repmat(data.stress_train,9,1);
[x,y] = meshgrid(data.x_train',data.y_train);
coord = [y(:), x(:)];
hist_init = zeros(length(coord),length(crackTipLoc)*8);

for iCrack = 1:length(crackTipLoc)   
    
    for iLoc = 1: length(coord)        
        if coord(iLoc,1) > crackTipLoc(iCrack,1)
            dis = sqrt((coord(iLoc,1)-crackTipLoc(iCrack,1))^2+(coord(iLoc,2)-crackTipLoc(iCrack,2))^2);
        else
            dis = abs(coord(iLoc,2)-crackTipLoc(iCrack,2));        
        end
        if dis <= l/2
            hist_init(iLoc,(iCrack-1)*8+1:iCrack*8) = constB*G*(1-dis/(l/2))/(2*l);
        end
    end
end
u_train = u_train(:,1:8);
v_train = v_train(:,1:8);
phi_train = phi_train(:,1:8);
hist_init = hist_init(:,1:8);
stress_train = stress_train(1:8,1);
save("dataset.mat","coord","hist_init",'u_train','v_train','phi_train','stress_train')
% scatter(coord(:,1),coord(:,2),[],v_train(:,8))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,2))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,3))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,4))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,5))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,6))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,7))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,8))
% scatter(coord(:,1),coord(:,2),[],hist_init(:,9))
