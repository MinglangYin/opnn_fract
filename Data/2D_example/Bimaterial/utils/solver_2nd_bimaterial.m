function solver_2nd_bimaterial(Input,PF,file_name,folder_name,iFacto)

% Reads the model and solver parameters from the input file
[geometry,Mater,Fract,Integ] = Input();
[PHTelem,controlPts,dimBasis] = PF.Geometry(geometry);
[PHTelem,sizeBasis] = zipConforming(PHTelem,dimBasis,geometry);

disp('Initializing boundary conditions on the initial geometry.')
[dirichlet] = PF.Boundary(PHTelem,geometry);

% cordinates = [controlPts{1}(:,1),controlPts{1}(:,2)];
% save('cordinates.mat','cordinates')
% constrainedPts = dirichlet.XY';
% constrainedDirection = dirichlet.ValXY;
% appliedDisp = dirichlet.restrainedPts;
% save('boundary.mat','constrainedPts','constrainedDirection','appliedDisp')

disp('Precomputing shape functions and derivatives.')
[shape,dgdx,volume,gaussCord,numberElements]=cartdev(PHTelem,controlPts,geometry);

disp('History function and phase field initialization.')
[fenerg] = PF.History(gaussCord,Fract,numberElements,geometry);

tdisp = zeros(3*sizeBasis,1);
solPhi = zeros(sizeBasis,1);
u = [];
v = [];
phi = [];

for istep = 1:Integ.nstep+1
    
    istep
    Integ.tfacto = Integ.tfacto + Integ.dfacto1;
    
    
    % Begin inner iteration
    normInnerStep = Inf;
    while (normInnerStep > geometry.toler)
        
        disp('Assembling the stiffness matrix.')
        tic
        [stiffUU,~] = PF.StiffUU(PHTelem,sizeBasis,numberElements,dgdx,...
            shape,Fract,gaussCord,Mater,volume,tdisp,geometry);
        
        disp('Imposing boundary conditions and solving.')
        solU = applyBoundary2D(dirichlet,Integ.tfacto,stiffUU,tdisp(1:2*sizeBasis));
        tdisp(1:2*sizeBasis) = solU;
        clear solU
        
        disp('Update the internal forces.')
        fenerg = internalForces_bimaterial(PHTelem,dgdx,tdisp,geometry,Mater,fenerg,gaussCord);
        
        disp('Updating phase field.')
        [stiffPhiPhi,RHSPhi] = PF.StiffPhiPhi(PHTelem,sizeBasis,numberElements,...
            dgdx,shape,Fract,Mater,volume,geometry,fenerg,gaussCord,tdisp);
        
        solPhiOld = solPhi;
        solPhi = stiffPhiPhi\RHSPhi;
        normInnerStep = norm(stiffPhiPhi*solPhiOld-RHSPhi)/norm(RHSPhi)
        tdisp(2*sizeBasis+1:end) = solPhi;
        clear stiffPhiPhi RHSPhi solPhiOld
        
    end
%         Phi = solPhi;
%         Phi(Phi<0) = 0;
%         Phi(Phi>1) = 1;
%         save('initPhi_short.mat','Phi')
    
    disp('Print data for force-tdisp curves.')
    PF.Trac(stiffUU,tdisp(1:2*sizeBasis),Integ.tfacto,dirichlet,file_name);
    if(mod(istep,Integ.nprint) == 0)% Print results
        u = [u, tdisp(1:2:2*sizeBasis)];
        v = [v, tdisp(2:2:2*sizeBasis)];
        phi = [phi, tdisp(2*sizeBasis+1:end)];
        plotDispPhase2D_bimaterial(PHTelem,tdisp,sizeBasis,numberElements,...
            geometry,controlPts,Mater,gaussCord,istep,folder_name)
    end %if
end %istep
save([folder_name,'/dFacto',num2str(iFacto),'.mat'], 'u','v','phi')
end