function c = apply_regularization(xi, reg_type, beta, backCond, c_min, N)
% APPLY_REGULARIZATION Apply regularization to the dual variable xi
%
% Inputs:
%   xi        - Dual variable (N x N matrix)
%   reg_type  - Regularization type: 'L1', 'L2', or 'TV'
%   beta      - Regularization strength parameter
%   backCond  - Background condition value (used for L1)
%   c_min     - Minimum value constraint for positivity
%   N         - Grid size
%
% Output:
%   c         - Regularized solution (N x N matrix)
%
% Examples:
%   c = apply_regularization(xi, 'TV', 10.0, 1, 0.5, 129);
%   c = apply_regularization(xi, 'L1', 0.1, 1, 0.5, 129);
%   c = apply_regularization(xi, 'L2', 10.0, 1, 0.5, 129);

switch reg_type
    case 'TV'
        % Total Variation regularization via PDHG
        % Solves: min_c ||c - xi||^2 + (1/beta) * TV(c)
        [c, ~] = TV_PDHG_host(zeros(N), zeros(N), xi, 1/beta, 200, 0);
        
    case 'L2'
        % L2 regularization (Tikhonov)
        % No explicit regularization step, just use xi directly
        c = xi;
        
    case 'L1'
        % L1 regularization (Soft thresholding / Shrinkage)
        % Proximal operator: prox_{beta * ||.||_1}(xi)
        tmp = xi - backCond;
        c = sign(tmp) .* max(abs(tmp) - beta, 0) + backCond;
        
    otherwise
        error('Unknown regularization type: %s. Use ''L1'', ''L2'', or ''TV''.', reg_type);
end

% Ensure positivity constraint
c = max(c, c_min);

end
