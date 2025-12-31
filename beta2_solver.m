function beta = beta2_solver(T, res, dx, dy)
% BETA2_SOLVER  Solve for the normalized adjoint weight beta (2D).
%
% We solve the transport equation (interior):
%     T_x * beta_x + T_y * beta_y = 0   in Omega
%
% Boundary condition:
%     (n · ∇T) * beta = res            on ∂Omega
% where res = T^* - T is the boundary residual (can be provided as a full
% matrix; only its boundary values are used).
% or res = J_r(T^* - T).
%
% Grid convention:
%   - i = row index (y-direction, spacing dy)
%   - j = col index (x-direction, spacing dx)
%   - T(i,j) is traveltime at grid point (x_j, y_i)
%
% Numerical method:
%   - Boundary: beta computed directly from BC
%   - Interior: upwind discretization + fast sweeping
%
% Inputs:
%   T    : traveltime field (I-by-J)
%   res  : residual field res = T_star - T (I-by-J), boundary values used
%   dx   : grid spacing in x (column direction)
%   dy   : grid spacing in y (row direction)
%
% Output:
%   beta : adjoint weight (I-by-J)

[I, J] = size(T);

% --------- input checks ----------
if ~isequal(size(res), [I, J])
    error("beta2_solver: size(res) must match size(T). Got (%d,%d) vs (%d,%d).", ...
        size(res,1), size(res,2), I, J);
end
if dx <= 0 || dy <= 0
    error("beta2_solver: dx and dy must be positive.");
end

% --------- parameters ----------
iterFS = 500;      % maximum fast sweeping iterations
tolFS  = 1e-10;    % convergence tolerance
epsg   = 1e-12;    % small value to protect division by zero

beta = zeros(I, J);

%% ---- Boundary Condition: (n · grad T) * beta = res ----
% Extract residuals on boundaries directly from res (= T_star - T)
rL = res(:,1);    % left   boundary (j=1)
rR = res(:,J);    % right  boundary (j=J)
rT = res(1,:);    % top    boundary (i=1)
rB = res(I,:);    % bottom boundary (i=I)

% Compute n · grad T on boundaries using one-sided differences
% Left:   n = (-1,0), n·∇T = -T_x ≈ -(T(:,2)-T(:,1))/dx
% Right:  n = (+1,0), n·∇T = +T_x ≈ +(T(:,J)-T(:,J-1))/dx
% Top:    n = (0,-1), n·∇T = -T_y ≈ -(T(2,:)-T(1,:))/dy
% Bottom: n = (0,+1), n·∇T = +T_y ≈ +(T(I,:)-T(I-1,:))/dy
gL = -(T(:,2)   - T(:,1))   / dx;
gR =  (T(:,J)   - T(:,J-1)) / dx;
gT = -(T(2,:)   - T(1,:))   / dy;
gB =  (T(I,:)   - T(I-1,:)) / dy;

% Protect against division by zero (keep sign)
gL = protect_denom(gL, epsg);
gR = protect_denom(gR, epsg);
gT = protect_denom(gT, epsg);
gB = protect_denom(gB, epsg);

% Boundary beta: beta = res / (n·∇T)
beta(:,1) = rL ./ gL;
beta(:,J) = rR ./ gR;
beta(1,:) = rT ./ gT;
beta(I,:) = rB ./ gB;

% Corners: average two adjacent boundary formulas
beta(1,1) = 0.5*(rL(1)/gL(1) + rT(1)/gT(1));
beta(I,1) = 0.5*(rL(I)/gL(I) + rB(1)/gB(1));
beta(1,J) = 0.5*(rR(1)/gR(1) + rT(J)/gT(J));
beta(I,J) = 0.5*(rR(I)/gR(I) + rB(J)/gB(J));

%% ---- Interior: Upwind Coefficients for T_x*beta_x + T_y*beta_y = 0 ----
% We compute one-sided derivatives of T (for upwind selection):
%   a_plus  = -T_y forward,  a_minus = -T_y backward
%   b_plus  = -T_x forward,  b_minus = -T_x backward

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

% Split into positive/negative parts for upwind formula
a_plus_p  = max(a_plus, 0);   a_plus_m  = min(a_plus, 0);
a_minus_p = max(a_minus, 0);  a_minus_m = min(a_minus, 0);
b_plus_p  = max(b_plus, 0);   b_plus_m  = min(b_plus, 0);
b_minus_p = max(b_minus, 0);  b_minus_m = min(b_minus, 0);

%% ---- Fast Sweeping Iteration ----
% Four sweep directions to follow characteristics.

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
            %   denom = sum of outgoing upwind coefficients
            denom = (a_plus_p(i,j) - a_minus_m(i,j))/dy ...
                  + (b_plus_p(i,j) - b_minus_m(i,j))/dx;

            if abs(denom) < 1e-14
                continue;  % degenerate points (e.g. near source/flat T)
            end

            %   num = contributions from upwind neighbors
            num = (a_minus_p(i,j)*beta(i-1,j) - a_plus_m(i,j)*beta(i+1,j))/dy ...
                + (b_minus_p(i,j)*beta(i,j-1) - b_plus_m(i,j)*beta(i,j+1))/dx;

            beta(i,j) = num / denom;
        end
    end

    % Convergence check (your original scaling kept)
    err = norm(beta(:) - beta_old(:)) * dx * dy;
    if err < tolFS
        break;
    end
end

end

%% ---- Helper Function ----
function g = protect_denom(g, epsg)
% PROTECT_DENOM  Replace near-zero values with signed epsilon.
mask = abs(g) < epsg;
if any(mask(:))
    s = sign(g(mask));
    s(s==0) = 1;
    g(mask) = s .* epsg;
end
end