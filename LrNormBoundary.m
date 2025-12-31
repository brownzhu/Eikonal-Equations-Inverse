function Lr = LrNormBoundary(T, T_star, dx, dy, r)
%LRNORMBOUNDARY  Compute ||T - T_star||_{L^r(∂Ω)} for r > 1 on a rectangular grid.
%
% Discrete boundary L^r norm (r>1):
%   ||e||_{L^r(∂Ω)}^r ≈  ∫_{∂Ω} |e|^r ds
% where e = T - T_star, and ds is approximated by dx or dy on each edge.
%
% Implementation detail (corner handling):
%   - Left/right edges: use rows 1:I-1 so corners are not double-counted
%   - Bottom/top edges: use cols 1:J-1 so corners are not double-counted
%
% Inputs:
%   T       : computed traveltime (I-by-J)
%   T_star  : target/measured traveltime (I-by-J) (only boundary used)
%   dx      : grid spacing in x-direction (columns)
%   dy      : grid spacing in y-direction (rows)
%   r       : exponent (scalar), must satisfy r > 1
%
% Output:
%   Lr      : discrete L^r(∂Ω) norm of (T - T_star)

    % -------- checks --------
    if nargin < 5
        error('LrNormBoundary requires inputs (T, T_star, dx, dy, r).');
    end
    if ~isequal(size(T), size(T_star))
        error('T and T_star must have the same size.');
    end
    if ~isscalar(r) || ~isfinite(r) || r <= 1
        error('r must be a finite scalar > 1.');
    end
    if dx <= 0 || dy <= 0
        error('dx and dy must be positive.');
    end

    [I, J] = size(T);

    % -------- boundary error e = T - T_star --------
    % Left/right edges (exclude last row to avoid double-counting corners)
    eL = T(1:I-1, 1) - T_star(1:I-1, 1);   % left boundary
    eR = T(1:I-1, J) - T_star(1:I-1, J);   % right boundary

    % Bottom/top edges (exclude last column to avoid double-counting corners)
    eT = T(1, 1:J-1) - T_star(1, 1:J-1);   % top boundary (i=1)
    eB = T(I, 1:J-1) - T_star(I, 1:J-1);   % bottom boundary (i=I)

    % -------- discrete integral of |e|^r over boundary --------
    % Left/right edges have arclength element ~ dy per grid step
    Er_lr = (sum(abs(eL).^r) + sum(abs(eR).^r)) * dy;

    % Top/bottom edges have arclength element ~ dx per grid step
    Er_tb = (sum(abs(eT).^r) + sum(abs(eB).^r)) * dx;

    % -------- L^r norm --------
    Lr = (Er_lr + Er_tb)^(1/r);
end