function Lr = LrNormBoundary3D(T, T_star, dx, dy, dz, r)
%LRNORMBOUNDARY3D  Compute ||T - T_star||_{L^r(∂Ω)} for r>1 on a 3D grid.
%
% Discrete approximation:
%   ||e||_{L^r(∂Ω)}^r ≈ ∫_{∂Ω} |e|^r dS
% where e = T - T_star.
%
% To avoid double-counting edges/corners, we use the same slicing strategy
% as your EnergyFun3D:
%   x-faces: i = 1 and i = I, with (j,k) in 1:J-1, 1:K-1
%   y-faces: j = 1 and j = J, with (i,k) in 1:I-1, 1:K-1
%   z-faces: k = 1 and k = K, with (i,j) in 1:I-1, 1:J-1
%
% Inputs:
%   T      : computed traveltime (I-by-J-by-K)
%   T_star : target traveltime (same size)
%   dx,dy,dz : grid spacings
%   r      : exponent, must satisfy r > 1
%
% Output:
%   Lr     : discrete L^r(∂Ω) norm of (T - T_star)

    % -------- checks --------
    if nargin < 6
        error('LrNormBoundary3D requires inputs (T, T_star, dx, dy, dz, r).');
    end
    if ~isequal(size(T), size(T_star))
        error('T and T_star must have the same size.');
    end
    if ~isscalar(r) || ~isfinite(r) || r <= 1
        error('r must be a finite scalar > 1.');
    end
    if dx <= 0 || dy <= 0 || dz <= 0
        error('dx, dy, dz must be positive.');
    end

    [I, J, K] = size(T);
    e = T - T_star;

    % -------- x = const faces (i = 1, I): area element dy*dz --------
    Ex = sum(abs(e([1, I], 1:J-1, 1:K-1)).^r, 'all') * dy * dz;

    % -------- y = const faces (j = 1, J): area element dx*dz --------
    Ey = sum(abs(e(1:I-1, [1, J], 1:K-1)).^r, 'all') * dx * dz;

    % -------- z = const faces (k = 1, K): area element dx*dy --------
    Ez = sum(abs(e(1:I-1, 1:J-1, [1, K])).^r, 'all') * dx * dy;

    % -------- L^r norm --------
    Lr = (Ex + Ey + Ez)^(1/r);
end