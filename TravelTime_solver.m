function T = TravelTime_solver(c, fixed_pt_list, dx, dy, I, J)
% TRAVELTIME_SOLVER  Solve the Eikonal equation using fast sweeping method.
%
% Solves:  |grad T| = 1/c   with T = 0 at source points
%
% Grid convention:
%   - i = row index, corresponds to y-direction, spacing dy
%   - j = column index, corresponds to x-direction, spacing dx
%   - T(i,j) is the value at grid point (x_j, y_i)
%
% Inputs:
%   c             : velocity field (I-by-J matrix)
%   fixed_pt_list : source points, each row is [val, j, i]
%                   where val=initial value (usually 0), j=column, i=row
%   dx            : grid spacing in x-direction (column direction)
%   dy            : grid spacing in y-direction (row direction)
%   I             : number of rows (y-direction)
%   J             : number of columns (x-direction)
%
% Output:
%   T             : traveltime field (I-by-J matrix)

% Backward compatibility: if dy is empty, use dx (square grid)
if isempty(dy)
    dy = dx;
end

T = ones(I, J)*999;
for idx = 1:size(fixed_pt_list, 1)
    val = fixed_pt_list(idx, 1);
    j = fixed_pt_list(idx, 2); 
    i = fixed_pt_list(idx, 3);
    T(i, j) = val;
end

interOrder = { [1, 1, I, 1, 1, J],...
    [1, 1, I, J, -1, 1],...
    [I, -1, 1, 1, 1, J],...
    [I, -1, 1, J, -1, 1]
};

interFS = 1000;
tolFS = 1e-6;
inter_erro = zeros(interFS, 1);
for k = 1: interFS
    
    T_temp = T;
    order = interOrder{cast(mod(k-1, 4), "uint8") + 1};
    for i = order(1): order(2): order(3)
        for j = order(4): order(5): order(6)     
            
            % y-direction (row/i): use dy
            if i ~= 1 && i ~= I
                a = min(T(i-1, j), T(i+1, j));
            else
                if i == 1
                    a = 2*T(i+1, j) - T(i+2, j);
                else
                    a = 2*T(i-1, j) - T(i-2, j);
                end
            end
            
            % x-direction (column/j): use dx
            if j ~= 1 && j ~= J
                b = min(T(i, j-1), T(i, j+1));
            else
                if j == 1
                    b = 2*T(i, j+1) - T(i, j+2);
                else
                    b = 2*T(i, j-1) - T(i, j-2);
                end
            end
            
            % Solve quadratic for Eikonal update with dx != dy
            % (T_x)^2 + (T_y)^2 = 1/c^2
            % Using upwind: ((T-a)/dy)^2 + ((T-b)/dx)^2 = 1/c^2
            slowness = 1 / c(i, j);
            
            if abs(a - b) >= slowness * dx * dy / sqrt(dx^2 + dy^2)
                % One-sided update: use the smaller neighbor
                if a < b
                    Tij = a + dy * slowness;
                else
                    Tij = b + dx * slowness;
                end
            else
                % Two-sided update: solve quadratic
                % (T-a)^2/dy^2 + (T-b)^2/dx^2 = 1/c^2
                % Let A = 1/dy^2 + 1/dx^2
                %     B = -2*(a/dy^2 + b/dx^2)
                %     C = a^2/dy^2 + b^2/dx^2 - 1/c^2
                A = 1/dy^2 + 1/dx^2;
                B = -2*(a/dy^2 + b/dx^2);
                C = a^2/dy^2 + b^2/dx^2 - slowness^2;
                discriminant = B^2 - 4*A*C;
                if discriminant >= 0
                    Tij = (-B + sqrt(discriminant)) / (2*A);
                else
                    % Fallback to one-sided
                    Tij = min(a + dy*slowness, b + dx*slowness);
                end
            end
            T(i, j) = min(T(i, j), Tij);
            
        end
    end
    % Reset source points
    for idx = 1:size(fixed_pt_list, 1)
        val = fixed_pt_list(idx, 1);
        j = fixed_pt_list(idx, 2); 
        i = fixed_pt_list(idx, 3);
        T(i, j) = val;
    end
    
    inter_erro(k) = norm(T - T_temp) * dx * dy;
    if inter_erro(k) < tolFS
        break;
    end

end