function ReducedBaseOffline(grid, N, ranges)
    
    rbpoints = random_log_points(ranges, N);

    % Reduced basis matrix Z
    disp('Starting calculation of reduced basis:');
    Z = zeros(grid.nodes, N);
    
    for i = 1:N
        params = [rbpoints(i, :)'; 1.0; 0.1];
        [Z(:, i), s] = ThermalFin(grid, params, 1);
        disp([i, rbpoints(i, :)]);
        fprintf('Output functional: %f\n', s);
    end

    % Preprocessing
    Q = 6;
    bb.ANq = zeros(Q, N, N);
    bb.FN = zeros(N, 1);
    bb.N = N;
    bb.Q = Q;

    for q = 1:Q
        ind = zeros(Q, 1);
        ind(q) = 1.0;
        [~, ~, A, f] = ThermalFin(grid, ind, 0);
        bb.ANq(q, :, :) = Z' * A * Z;
    end
    bb.FN = Z' * f;

    % Save to file
    save('ReducedBasisData.mat', 'bb');
end

function [points] = random_log_points(ranges, N)
    epsilon = 0.01;

    % Precompute
    ran_min = log(ranges(:, 1) + epsilon);
    ran_max = log(ranges(:, 2) + epsilon);
    ran_diff = ran_max - ran_min;
    rng(166214);
    points = ran_min' + ran_diff' .* rand(N, size(ranges, 1));
    
    % Exponentiate and adjust
    points = exp(points) - epsilon;
end

