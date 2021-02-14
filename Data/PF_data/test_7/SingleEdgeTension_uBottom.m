% Script for a two dimensional plate under tension

close all
clear

addpath('../utils')
addpath('../example_data')
addpath('../../nurbs/inst')

dfactoSteps = [1e-05, 1.1e-04, 4.2e-04, 6.1e-04, 9.1e-04, 1e-03];
curDir = './';
for iFacto = 1: length(dfactoSteps)
    
    folder_name = [curDir 'dFacto' num2str(iFacto)];%new folder name
    mkdir(folder_name)
    dfacto = dfactoSteps(iFacto);
    Input = @() Input_tensilePlate(dfacto);
    PF.Geometry = @(geometry)initMeshTensile(geometry);
    PF.Boundary = @(PHTelem,geometry,controlPts)initialBC_tensile(PHTelem,geometry);
    PF.History = @(gaussCord,Fract,numberElements,geometry)history_tensile_uBottom(gaussCord,...
        Fract,numberElements,geometry);
    PF.Trac = @(stiffUU,tdisp,Integ,dirichlet,file_name)compTreacTension(stiffUU,tdisp,...
        Integ,dirichlet,file_name);
    
    file_name = [folder_name,'FD-tensile.txt'];
    output = fopen(file_name,'w');
    fprintf(output,'%14.6e %14.6e\n',0,0);
    fclose(output);  
    
    PF.StiffUU = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,Mater,...
        volume,tdisp,geometry)gStiffnessUU(PHTelem,sizeBasis,...
        numberElements,dgdx,shape,Mater,volume,tdisp,geometry);
    PF.StiffPhiPhi = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,...
        Mater,volume,geometry,fenerg,tdisp)gStiffnessPhiPhi(PHTelem,...
        sizeBasis,numberElements,dgdx,shape,Fract,Mater,volume,geometry,fenerg);
    
    solver_2nd(Input,PF,file_name,folder_name,iFacto);
    
end
