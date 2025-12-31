function c = apply_regularization_ZM(xi, reg_type, beta, backCond, c_min, ~)
% APPLY_REGULARIZATION Apply regularization to the dual variable xi
%
% Inputs:
%   xi        - Dual variable (Nz x Nx matrix, can be non-square)
%   reg_type  - Regularization type: 'L1', 'L2', or 'TV'
%   beta      - Regularization strength parameter
%   backCond  - Background condition value (used for L1)
%   c_min     - Minimum value constraint for positivity
%   ~         - (unused, kept for backward compatibility)
%
% Output:
%   c         - Regularized solution (same size as xi)
%
% Examples:
%   c = apply_regularization(xi, 'TV', 10.0, 1, 0.5);
%   c = apply_regularization(xi, 'L1', 0.1, 1, 0.5);
%   c = apply_regularization(xi, 'L2', 10.0, 1, 0.5);

% Get actual size of xi (supports non-square grids)
[Nz, Nx] = size(xi);

switch reg_type
    case 'TV'
        % Total Variation regularization via PDHG
        % Solves: min_c 1/2||c - beta xi||^2 + beta * TV(c)
        [c, ~] = TV_PDHG_host(zeros(Nz, Nx), zeros(Nz, Nx), beta*xi, 1/beta, 200, 0);
        
    case 'L2'
        % L2 regularization (Tikhonov)
        % No explicit regularization step, just use xi directly
        c = beta*xi;
        
    case 'L1'
        % L1 regularization (Soft thresholding / Shrinkage)
        % Proximal operator: prox_{beta * ||.||_1}(xi)
        tmp = beta*xi - backCond;
        c = sign(tmp) .* max(abs(tmp) - beta, 0) + backCond;
        
    otherwise
        error('Unknown regularization type: %s. Use ''L1'', ''L2'', or ''TV''.', reg_type);
end

% Ensure positivity constraint
c = max(c, c_min);

end
