load('grids.mat');  
Th = fine; % Options are coarse, medium, fine meshes
params = [0.4; 0.6; 0.8; 1.2; 1.0; 0.1];  % Column vector

try
    [uh_mu0, Troot_h_mu0, A, f] = ThermalFin(Th, params, 1);
    plotsolution(Th, uh_mu0);
    disp('The value of Troot_h for µ0 is:');
    disp(Troot_h_mu0);
catch ME
    disp('An error occurred:');
    disp(ME.message);
end
