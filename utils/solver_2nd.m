function solver_2nd(Input,PF,file_name,folder_name,iFacto)

% Reads the model and solver parameters from the input file
[geometry,Mater,Fract,Integ] = Input();
[PHTelem,controlPts,dimBasis] = PF.Geometry(geometry);
[PHTelem,sizeBasis] = zipConforming(PHTelem,dimBasis,geometry);

disp('Initializing boundary conditions on the initial geometry.')
[dirichlet] = PF.Boundary(PHTelem,geometry,controlPts);

disp('Precomputing shape functions and derivatives.')
[shape,dgdx,volume,gaussCord,numberElements] = cartdev(PHTelem,controlPts,geometry);

disp('History function and phase field initialization.')
Fract.constl = 2*Fract.constl;
[fenerg] = PF.History(gaussCord,Fract,numberElements,geometry);
Fract.constl = Fract.constl/2;
clear gaussCord

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
    miter = 0;
    while (normInnerStep > geometry.toler)
        refFlag = 0;
        disp('Assembling the stiffness matrix.')
        tic
        [stiffUU,elemRef] = PF.StiffUU(PHTelem,sizeBasis,numberElements,...
            dgdx,shape,Fract,Mater,volume,tdisp,geometry);
        toc
        
        disp('Imposing boundary conditions and solving.')
        tic
        solU = applyBoundary2D(dirichlet,Integ.tfacto,stiffUU,tdisp(1:2*sizeBasis));
        tdisp(1:2*sizeBasis) = solU;
        clear solU
        toc
        
        disp('Update the internal forces.')
        fenerg = internalForces(PHTelem,dgdx,tdisp,geometry,Mater,fenerg);
        
        disp('Updating phase field.')
        tic
        [stiffPhiPhi,RHSPhi] = PF.StiffPhiPhi(PHTelem,sizeBasis,numberElements,...
            dgdx,shape,Fract,Mater,volume,geometry,fenerg,tdisp);
        solPhiOld = solPhi;
        solPhi = stiffPhiPhi\RHSPhi;
        normInnerStep = norm(stiffPhiPhi*solPhiOld-RHSPhi)/norm(RHSPhi)
        miter = miter + 1
        tdisp(2*sizeBasis+1:end) = solPhi;
        clear stiffPhiPhi RHSPhi solPhiOld
        
        [solPhiPatch] = transferFieldGlob2Loc(PHTelem,dimBasis,solPhi);
        toc
        for iPatch = 1:geometry.numPatches
            if sum(elemRef{iPatch})>0
                % Refine and update the mesh
                refFlag = 1;
                disp(['In patch ',num2str(iPatch),' refining ',num2str(sum(elemRef{iPatch})),' elements.'])
                [PHTelem{iPatch},controlPts{iPatch},dimBasis(iPatch),solPhiPatch{iPatch}, ...
                    numberElements] = refineElemProjGradedIso(elemRef{iPatch},PHTelem{iPatch}, ...
                    controlPts{iPatch},geometry,dimBasis(iPatch),solPhiPatch{iPatch},numberElements);
            end
        end
        
        if refFlag
            tic
            clear stiffUU
            [PHTelem,sizeBasis] = zipConforming(PHTelem,dimBasis,geometry);
            [dirichlet] = PF.Boundary(PHTelem,geometry,controlPts);
            [shape,dgdx,volume,gaussCord,numberElements]=cartdev(PHTelem,controlPts,geometry);
            [fenerg] = PF.History(gaussCord,Fract,numberElements,geometry);
            clear gaussCord
            
            solPhi = transferFieldLoc2Glob(PHTelem,sizeBasis,solPhiPatch);
            tdisp = zeros(3*sizeBasis,1); % Solution Vector
            tdisp(2*sizeBasis+1:end) = solPhi;
            normInnerStep = Inf;
            toc
        end
    end
    
    disp('Print data for force-tdisp curves.')
    tic
    PF.Trac(stiffUU,tdisp(1:2*sizeBasis),Integ.tfacto,dirichlet,file_name);
    if(mod(istep,Integ.nprint) == 0)% Print results
        u = [u, tdisp(1:2:2*sizeBasis)];
        v = [v, tdisp(2:2:2*sizeBasis)];
        phi = [phi, tdisp(2*sizeBasis+1:end)];
        fprintf('Done step: %5d\n',istep);
        plotDispPhase2D(PHTelem,tdisp,sizeBasis,numberElements,geometry,controlPts,Mater,istep,folder_name);
    end %if
    toc
end %istep
save([folder_name,'/dFacto',num2str(iFacto),'.mat'], 'u','v','phi')
end