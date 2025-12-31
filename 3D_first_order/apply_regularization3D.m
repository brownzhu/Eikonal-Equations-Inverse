function [c, tv_state] = apply_regularization3D(xi, reg_type, beta, backCond, c_min, opts)
%APPLY_REGULARIZATION3D  Apply regularization to a 3D field xi.
%
% This function maps the "dual-like" variable xi to a physical parameter c
% through a regularization/prox step, restricted to 3D arrays.
%
% Supported reg_type:
%   'L2' : c = xi
%   'L1' : soft-thresholding around backCond with threshold beta
%   'TV' : TV proximal step (3D ROF denoising)
%          c = argmin_c 0.5||c - xi||_2^2 + beta * TV(c)
%
% TV branch uses TV_PDHD_3D which solves the ROF form:
%   min_u  TV(u) + (lbd/2) * ||u - f||_2^2
% To match prox_{beta*TV}(xi), set:
%   f = xi,   lbd = 1/beta   (beta>0)
%
% Inputs:
%   xi        : 3D array (I-by-J-by-K)
%   reg_type  : 'L1', 'L2', or 'TV'
%   beta      : regularization strength (beta>=0; for TV need beta>0)
%   backCond  : background value (only used for L1)
%   c_min     : positivity lower bound
%   opts      : struct (optional), fields:
%              opts.tv_maxit   (default 300)
%              opts.tv_gaptol  (default 1e-4)
%              opts.tv_verbose (default 0)  % (TV_PDHD_3D prints if it has verbose=1 internally)
%              opts.tv_warmstart (default false)
%              opts.w1, opts.w2, opts.w3 (warmstart dual variables if tv_warmstart=true)
%
% Outputs:
%   c        : regularized 3D result, same size as xi
%   tv_state : struct with TV solver state (returned for warm-starting):
%              tv_state.w1, tv_state.w2, tv_state.w3

if nargin < 6 || isempty(opts)
    opts = struct();
end
if ~isfield(opts,'tv_maxit');      opts.tv_maxit = 300;  end
if ~isfield(opts,'tv_gaptol');     opts.tv_gaptol = 1e-4; end
if ~isfield(opts,'tv_warmstart');  opts.tv_warmstart = false; end

% ---- enforce 3D only ----
if ndims(xi) ~= 3
    error('apply_regularization3D: xi must be a 3D array (I-by-J-by-K).');
end

if ~isscalar(beta) || ~isfinite(beta) || beta < 0
    error('apply_regularization3D: beta must be a finite scalar >= 0.');
end
if ~isscalar(c_min) || ~isfinite(c_min)
    error('apply_regularization3D: c_min must be a finite scalar.');
end

tv_state = struct('w1', [], 'w2', [], 'w3', []);

switch upper(reg_type)

    case 'TV'
        % ===== TV prox =====
        % prox_{beta TV}(xi) = argmin_c 0.5||c-xi||^2 + beta TV(c)
        % ROF form used by TV_PDHD_3D:
        %   min_u TV(u) + (lbd/2)||u - f||^2
        % Match by: f = xi, lbd = 1/beta.
        if beta == 0
            c = xi;  % no TV penalty
        else
            f   = xi;
            lbd = 1 / beta;

            % Initialize dual variables (w1,w2,w3) for TV_PDHD_3D
            if opts.tv_warmstart && isfield(opts,'w1') && isfield(opts,'w2') && isfield(opts,'w3') ...
                               && isequal(size(opts.w1),size(xi)) && isequal(size(opts.w2),size(xi)) && isequal(size(opts.w3),size(xi))
                w1 = opts.w1; w2 = opts.w2; w3 = opts.w3;
            else
                w1 = zeros(size(xi));
                w2 = zeros(size(xi));
                w3 = zeros(size(xi));
            end

            % Solve TV-ROF (3D)
            [c, w1, w2, w3, ~, ~, ~, ~] = TV_PDHD_3D(w1, w2, w3, f, lbd, opts.tv_maxit, opts.tv_gaptol);

            % Return state for possible warmstart outside
            tv_state.w1 = w1; tv_state.w2 = w2; tv_state.w3 = w3;
        end

    case 'L2'
        % ===== L2 =====
        c = xi;

    case 'L1'
        % ===== L1 prox =====
        % prox_{beta||.||_1}(xi) applied to (xi-backCond), then shift back.
        tmp = xi - backCond;
        c = sign(tmp) .* max(abs(tmp) - beta, 0) + backCond;

    otherwise
        error("apply_regularization3D: Unknown reg_type '%s'. Use 'L1','L2','TV'.", reg_type);
end

% ---- positivity constraint ----
c = max(c, c_min);

end