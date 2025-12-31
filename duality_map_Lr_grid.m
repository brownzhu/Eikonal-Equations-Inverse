function [Jx, info] = duality_map_Lr_grid(x, r, dx, dy, dz)
%DUALITY_MAP_LR_GRID  Duality mapping for discrete L^r on FD grids (r>1).
%
% This function computes a duality mapping J_r(x) consistent with the choice:
%
%   Discrete L^r "integral-like" norm:
%       ||x||_{r,h}^r = sum_i |x_i|^r * w
%   Dual pairing (dot product WITHOUT weight):
%       <y, x> = sum_i y_i * x_i
%
% Under this convention, to satisfy the defining property
%       <J_r(x), x> = ||x||_{r,h}^r,
% we must include the cell measure w in the mapping:
%
%   (J_r(x))_i = w * |x_i|^{r-2} x_i
%              = w * sign(x_i) * |x_i|^{r-1}.
%
% Notes:
%  - r must be > 1 (so the mapping is single-valued and smooth away from 0).
%  - Works for both 2D arrays (I-by-J) and 3D arrays (I-by-J-by-K).
%  - If you instead use WEIGHTED pairing <y,x>_h = sum_i y_i x_i w,
%    then the correct mapping would be J_r(x)=|x|^{r-2}x (WITHOUT w).
%
% Inputs:
%   x   : grid function (2D or 3D numeric array)
%   r   : exponent (scalar), must satisfy r > 1
%   dx  : grid spacing in x-direction
%   dy  : grid spacing in y-direction
%   dz  : (optional) grid spacing in z-direction for 3D; omit for 2D
%
% Outputs:
%   Jx   : duality mapping array, same size as x
%   info : struct with metadata and quick checks:
%          info.dim        : 2 or 3
%          info.w          : cell measure (dx*dy or dx*dy*dz)
%          info.q          : conjugate exponent q = r/(r-1)
%          info.pair       : sum(Jx(:).*x(:))      (unweighted pairing)
%          info.norm_r_pow : info.w*sum(abs(x(:)).^r)  (||x||_{r,h}^r)
%          info.rel_err    : relative mismatch between pair and norm_r_pow

    % ---------- argument checks ----------
    if nargin < 4
        error('Need at least x, r, dx, dy.');
    end
    if ~isscalar(r) || ~isfinite(r) || r <= 1
        error('r must be a finite scalar > 1.');
    end
    if ~isscalar(dx) || ~isscalar(dy) || dx <= 0 || dy <= 0
        error('dx and dy must be positive scalars.');
    end

    % Determine dimension (2D vs 3D)
    if ndims(x) == 2
        dim = 2;
        w = dx * dy;
        if nargin >= 5 && ~isempty(dz)
            warning('dz provided but x is 2D; dz will be ignored.');
        end
    elseif ndims(x) == 3
        dim = 3;
        if nargin < 5 || isempty(dz)
            error('For 3D x, you must provide dz.');
        end
        if ~isscalar(dz) || dz <= 0
            error('dz must be a positive scalar for 3D.');
        end
        w = dx * dy * dz;
    else
        error('x must be a 2D or 3D numeric array.');
    end

    % Conjugate exponent
    q = r / (r - 1);

    % ---------- compute duality mapping ----------
    % Jx = w * sign(x) .* |x|^(r-1)
    % Using abs(x).^(r-2).*x is equivalent and avoids sign(0) ambiguity.
    Jx = w * (abs(x).^(r - 2)) .* x;

    % ---------- diagnostics (optional but useful) ----------
    pair = sum(Jx(:) .* x(:));            % <Jx, x>  (unweighted)
    norm_r_pow = w * sum(abs(x(:)).^r);   % ||x||_{r,h}^r
    denom = max(1e-30, abs(norm_r_pow) + abs(pair));
    rel_err = abs(pair - norm_r_pow) / denom;

    info = struct();
    info.dim = dim;
    info.w = w;
    info.q = q;
    info.pair = pair;
    info.norm_r_pow = norm_r_pow;
    info.rel_err = rel_err;
end