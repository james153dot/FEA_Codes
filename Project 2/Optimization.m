% Load the precomputed reduced-basis data
load('ReducedBasisData.mat');
k1 = 0.4; k2 = 0.6; k3 = 0.8; k4 = 1.2; k0 = 1.0;
costFunction = @(Bi) 0.2 * Bi + ReducedBaseOnline([k1, k2, k3, k4, k0, Bi], bb);

% Use fminbnd to find the optimal Bi within the range [0.1, 10]
optimalBi = fminbnd(costFunction, 0.1, 10);
minCost = costFunction(optimalBi);

disp(['Optimal Bi: ', num2str(optimalBi)]);
disp(['Minimum Cost: ', num2str(minCost)]);
