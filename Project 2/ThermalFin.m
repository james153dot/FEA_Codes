function [u, s, A, f] = ThermalFin(grid, params, solve)
    % Thank you office hours: could not have done this myself
    % Initialization
    A = spalloc(grid.nodes, grid.nodes, 10 * grid.nodes);
    f = zeros(grid.nodes, 1);

    % Stiffness summation
    ind = find(abs(params(1:5)) > eps);
    for i = ind'
        for j = (grid.theta{i})'
            % Node coordinates
            x1 = grid.coor(j(1), 1);
            y1 = grid.coor(j(1), 2);
            x2 = grid.coor(j(2), 1);
            y2 = grid.coor(j(2), 2);
            x3 = grid.coor(j(3), 1);
            y3 = grid.coor(j(3), 2);
            
            % Triangle Area
            area = -x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3;
            
            % Compute coefficients for the local stiffness matrix
            c11 = y2 - y3; 
            c12 = y3 - y1; 
            c13 = y1 - y2;
            c21 = x3 - x2; 
            c22 = x1 - x3; 
            c23 = x2 - x1;
            
            % Local stiffness matrix
            S1 = params(i) / (2 * abs(area)) * ...
                 [c11 * c11 + c21 * c21, c11 * c12 + ...
                 c21 * c22, c11 * c13 + c21 * c23;
                  c12 * c11 + c22 * c21, c12 * c12 + ...
                  c22 * c22, c12 * c13 + c22 * c23;
                  c13 * c11 + c23 * c21, c13 * c12 + ...
                  c23 * c22, c13 * c13 + c23 * c23];
            
            % Global stiffness matrix
            A(j, j) = A(j, j) + S1;
        end
    end

    % Robin B.C.
    if abs(params(6)) > 0
        for j = (grid.theta{6})'
            % Node coordinates
            x1 = grid.coor(j(1), 1);
            y1 = grid.coor(j(1), 2);
            x2 = grid.coor(j(2), 1);
            y2 = grid.coor(j(2), 2);
            dx = sqrt((x2 - x1)^2 + (y2 - y1)^2);
            
            % Robin boundary
            mass = params(6) * [dx / 3, dx / 6; dx / 6, dx / 3];
            
            % Global stiffness matrix
            A(j, j) = A(j, j) + mass;
        end
    end

    % Neumann B.C.
    for j = (grid.theta{7})'
        % Node coordinates
        x1 = grid.coor(j(1), 1);
        y1 = grid.coor(j(1), 2);
        x2 = grid.coor(j(2), 1);
        y2 = grid.coor(j(2), 2);
        dx = sqrt((x2 - x1)^2 + (y2 - y1)^2);
        
        % Update
        f(j) = f(j) + [dx / 2; dx / 2];
    end

    % Initialization
    u = 0.0;
    s = 0.0;

    % Solver
    if solve ~= 0
        u = A \ f;
        s = dot(u, f);
    end
end
