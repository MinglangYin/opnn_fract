%  Script for a two dimensional plate made of 2 materials subjected to tension

close all
clear

addpath('../utils')
addpath('../example_data')
addpath('../../../nurbs/inst')

dfactoSteps = [1e-04, 7.1e-04, 1.1e-03, 3.1e-03, 1e-02];
curDir = pwd;
for iFacto = 1: length(dfactoSteps)
    
    folder_name = [curDir '/dFacto' num2str(iFacto)];% New folder name
    mkdir(folder_name)
    dfacto = dfactoSteps(iFacto);
    
    Input = @() Input_bimaterial(dfacto);
    PF.Geometry = @(geometry)initMeshTensile(geometry);
    PF.Boundary = @(PHTelem,geometry)initialBC_bimaterial(PHTelem,geometry);
    PF.History = @(gaussCord,Fract,numberElements,geometry)history_bimaterial_hori(gaussCord,...
        Fract,numberElements,geometry);
    PF.Trac = @(stiffUU,tdisp,Integ,dirichlet,file_name)compTreacTension(stiffUU,tdisp,...
        Integ,dirichlet,file_name);
    
    file_name = [folder_name,'/FD-bimaterial.txt'];
    output = fopen(file_name,'w');
    fprintf(output,'%14.6e %14.6e\n',0,0);
    fclose(output);   
    
    PF.StiffUU = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,gaussCord,Mater,...
        volume,tdisp,geometry)gStiffnessUU_bimaterial(PHTelem,sizeBasis,...
        numberElements,dgdx,shape,gaussCord,Mater,volume,tdisp,geometry);
    
    PF.StiffPhiPhi = @(PHTelem,sizeBasis,numberElements,dgdx,shape,Fract,...
        Mater,volume,geometry,fenerg,gaussCord,tdisp)gStiffnessPhiPhi_bimaterial(PHTelem,...
        sizeBasis,numberElements,dgdx,shape,Fract,Mater,volume,geometry,fenerg,gaussCord);
    
    solver_2nd_bimaterial(Input,PF,file_name,folder_name,iFacto);
    
end
