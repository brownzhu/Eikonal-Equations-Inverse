function beta = beta_solver(T, T_star, dx, dy)
% BETA_SOLVER  Solve for the normalized adjoint weight beta.
%
% Solves the following transport equation using fast sweeping method:
%
%   PDE (interior):
%       T_x * beta_x + T_y * beta_y = 0    in Omega
%
%   BC (boundary):
%       (n · grad T) * beta = T_star - T   on ∂Omega
%
% where n is the outward unit normal on the boundary.
%
% Grid convention:
%   - i = row index, corresponds to y-direction, spacing dy
%   - j = column index, corresponds to x-direction, spacing dx
%   - T(i,j) is the value at grid point (x_j, y_i)
%
% Numerical method:
%   - Boundary: Direct evaluation from BC
%   - Interior: Upwind finite difference with fast sweeping iteration
%
% Inputs:
%   T       : traveltime field (I-by-J matrix)
%   T_star  : measured/target traveltime (same size as T, only boundary used)
%   dx      : grid spacing in x-direction (column/j direction)
%   dy      : grid spacing in y-direction (row/i direction)
%
% Output:
%   beta    : normalized adjoint weight (I-by-J matrix)
%
% See also: lambda_solver, TravelTime_solver

[I, J] = size(T);
if ~isequal(size(T_star), [I, J])
    error("beta_solver: size(T_star) must match size(T). Got (%d,%d) vs (%d,%d).", ...
        size(T_star,1), size(T_star,2), I, J);
end

iterFS = 500;      % maximum fast sweeping iterations
tolFS  = 1e-10;    % convergence tolerance
epsg   = 1e-12;    % small value to protect division by zero

beta = zeros(I, J);

%% ---- Boundary Condition: (n · grad T) * beta = T_star - T ----
% Compute residuals r = T_star - T on each boundary
rL = T_star(:,1) - T(:,1);    % left   (j=1)
rR = T_star(:,J) - T(:,J);    % right  (j=J)
rT = T_star(1,:) - T(1,:);    % top    (i=1)
rB = T_star(I,:) - T(I,:);    % bottom (i=I)

% Compute n · grad T on each boundary using one-sided difference
% Left:   n = (-1,0), n·∇T = -T_x ≈ -(T(:,2)-T(:,1))/dx
% Right:  n = (+1,0), n·∇T = +T_x ≈ +(T(:,J)-T(:,J-1))/dx
% Top:    n = (0,-1), n·∇T = -T_y ≈ -(T(2,:)-T(1,:))/dy
% Bottom: n = (0,+1), n·∇T = +T_y ≈ +(T(I,:)-T(I-1,:))/dy
gL = -(T(:,2)   - T(:,1))   / dx;
gR =  (T(:,J)   - T(:,J-1)) / dx;
gT = -(T(2,:)   - T(1,:))   / dy;
gB =  (T(I,:)   - T(I-1,:)) / dy;

% Protect against division by zero
gL = protect_denom(gL, epsg);
gR = protect_denom(gR, epsg);
gT = protect_denom(gT, epsg);
gB = protect_denom(gB, epsg);

% Compute beta on boundaries: beta = r / g
beta(:,1) = rL ./ gL;
beta(:,J) = rR ./ gR;
beta(1,:) = rT ./ gT;
beta(I,:) = rB ./ gB;

% Corners: average contributions from two adjacent boundaries
beta(1,1) = 0.5*(rL(1)/gL(1) + rT(1)/gT(1));
beta(I,1) = 0.5*(rL(I)/gL(I) + rB(1)/gB(1));
beta(1,J) = 0.5*(rR(1)/gR(1) + rT(J)/gT(J));
beta(I,J) = 0.5*(rR(I)/gR(I) + rB(J)/gB(J));

%% ---- Interior: Upwind Coefficients ----
% Discretize T_x * beta_x + T_y * beta_y = 0 using upwind scheme.
%
% Define one-sided derivatives of T (with sign for upwind formulation):
%   a_plus  = -T_y^+  (forward diff in i-direction)
%   a_minus = -T_y^-  (backward diff in i-direction)
%   b_plus  = -T_x^+  (forward diff in j-direction)
%   b_minus = -T_x^-  (backward diff in j-direction)
%
% Index convention:
%   i = row = y-direction → use dy
%   j = col = x-direction → use dx

a_plus  = zeros(I, J);
a_minus = zeros(I, J);
b_plus  = zeros(I, J);
b_minus = zeros(I, J);

for i = 2:I-1
    for j = 2:J-1
        a_plus(i, j)  = -(T(i+1, j) - T(i, j)) / dy;  % -T_y forward
        a_minus(i, j) = -(T(i, j) - T(i-1, j)) / dy;  % -T_y backward
        b_plus(i, j)  = -(T(i, j+1) - T(i, j)) / dx;  % -T_x forward
        b_minus(i, j) = -(T(i, j) - T(i, j-1)) / dx;  % -T_x backward
    end
end

% Split into positive and negative parts for upwind scheme
a_plus_p  = max(a_plus, 0);   a_plus_m  = min(a_plus, 0);
a_minus_p = max(a_minus, 0);  a_minus_m = min(a_minus, 0);
b_plus_p  = max(b_plus, 0);   b_plus_m  = min(b_plus, 0);
b_minus_p = max(b_minus, 0);  b_minus_m = min(b_minus, 0);

%% ---- Fast Sweeping Iteration ----
% Solve T_x * beta_x + T_y * beta_y = 0 in interior points.
% Four alternating sweep directions to capture all characteristic directions.

interOrder = {
    [2, 1, I-1, 2, 1, J-1],...    % i: 2→I-1, j: 2→J-1
    [2, 1, I-1, J-1, -1, 2],...   % i: 2→I-1, j: J-1→2
    [I-1, -1, 2, 2, 1, J-1],...   % i: I-1→2, j: 2→J-1
    [I-1, -1, 2, J-1, -1, 2]      % i: I-1→2, j: J-1→2
};

for k = 1:iterFS
    beta_old = beta;

    order = interOrder{mod(k-1,4)+1};
    for i = order(1):order(2):order(3)
        for j = order(4):order(5):order(6)

            % Upwind discretization:
            %   denom = sum of absolute upwind coefficients
            %   num   = sum of upwind neighbor contributions
            denom = (a_plus_p(i,j) - a_minus_m(i,j))/dy ...
                  + (b_plus_p(i,j) - b_minus_m(i,j))/dx;
            
            if abs(denom) < 1e-14
                continue;  % skip degenerate points (e.g., source)
            end

            num = (a_minus_p(i,j)*beta(i-1,j) - a_plus_m(i,j)*beta(i+1,j))/dy ...
                + (b_minus_p(i,j)*beta(i,j-1) - b_plus_m(i,j)*beta(i,j+1))/dx;

            beta(i,j) = num / denom;
        end
    end

    % Check convergence
    err = norm(beta(:) - beta_old(:)) * dx * dy;
    if err < tolFS
        break;
    end
end

end

%% ---- Helper Function ----
function g = protect_denom(g, epsg)
% PROTECT_DENOM  Replace near-zero values with signed epsilon.
%   Prevents division by zero while preserving sign information.
mask = abs(g) < epsg;
if any(mask(:))
    g_masked = g(mask);
    s = sign(g_masked);
    s(s==0) = 1;          % default to positive if exactly zero
    g(mask) = s .* epsg;
end
end