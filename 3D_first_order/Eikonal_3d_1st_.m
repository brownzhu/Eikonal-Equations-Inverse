% =========================
% File: Eikonal_3d_1st_.m  (Optimized Route-A)
% =========================
function [tau, info] = Eikonal_3d_1st_(c, src_idx, dx, dy, dz, opts)
% EIKONAL_3D_1ST_  3D isotropic Eikonal solver (1st-order upwind, fast sweeping).
%   Solve: |grad tau| = 1/c, with tau(src)=0.
%
% Inputs:
%   c       : (I,J,K) velocity field, must be positive
%   src_idx : [i,j,k] (1-based indices)
%   dx,dy,dz: grid spacing
%   opts    : struct options (see defaults below)
%
% Outputs:
%   tau  : travel time
%   info : struct (iters, converged, maxd_hist, warnings, etc.)

    % ---------- defaults ----------
    if nargin < 6, opts = struct(); end
    opts = set_default(opts, 'max_sweeps', 200);
    opts = set_default(opts, 'tol', 1e-6);
    opts = set_default(opts, 'large', 1e9);
    opts = set_default(opts, 'debug', false);
    opts = set_default(opts, 'debug_every', 1);
    opts = set_default(opts, 'warn_limit', 20);
    opts = set_default(opts, 'check_finite', true);
    opts = set_default(opts, 'skip_source_update', true);

    info = struct();
    info.converged  = false;
    info.iters      = 0;
    info.maxd_hist  = [];
    info.warn_count = 0;

    % ---------- basic checks ----------
    [I,J,K] = size(c);
    si = src_idx(1); sj = src_idx(2); sk = src_idx(3);

    if opts.debug
        fprintf('[Eikonal_3d_1st_] grid=(%d,%d,%d), dx=%.4g dy=%.4g dz=%.4g\n', I,J,K,dx,dy,dz);
        fprintf('[Eikonal_3d_1st_] source index=(%d,%d,%d)\n', si,sj,sk);
    end

    if any(src_idx < 1) || si>I || sj>J || sk>K
        error('src_idx out of bounds.');
    end
    if ~(isfinite(dx) && isfinite(dy) && isfinite(dz) && dx>0 && dy>0 && dz>0)
        error('dx,dy,dz must be positive finite.');
    end
    if any(~isfinite(c(:)))
        error('Velocity c contains nonfinite values.');
    end
    if any(c(:) <= 0)
        if opts.debug
            fprintf('[WARN] c has non-positive entries. Clamping to eps.\n');
        end
        c = max(c, eps);
    end

    % ---------- precompute constants ----------
    s = 1.0 ./ c;              % slowness
    invdx2 = 1.0 / (dx*dx);
    invdy2 = 1.0 / (dy*dy);
    invdz2 = 1.0 / (dz*dz);

    % ---------- init tau ----------
    large = opts.large;
    tau = large * ones(I,J,K);
    tau(si,sj,sk) = 0;

    % ---------- padded tau (avoid boundary checks) ----------
    % tau_pad size: (I+2, J+2, K+2), outer layer = large
    tau_pad = large * ones(I+2, J+2, K+2);
    tau_pad(2:I+1, 2:J+1, 2:K+1) = tau;

    % 8 sweep directions (ranges only)
    dirs = {
        1:I,    1:J,     1:K
        1:I,    1:J,     K:-1:1
        1:I,    J:-1:1,  1:K
        1:I,    J:-1:1,  K:-1:1
        I:-1:1, 1:J,     1:K
        I:-1:1, 1:J,     K:-1:1
        I:-1:1, J:-1:1,  1:K
        I:-1:1, J:-1:1,  K:-1:1
    };

    % ---------- sweeps ----------
    for sweep = 0:opts.max_sweeps
        t0 = tic;
        maxd = 0;

        for dd = 1:8
            ir = dirs{dd,1};
            jr = dirs{dd,2};
            kr = dirs{dd,3};

            for k = kr
                kp = k + 1; % padded index
                for j = jr
                    jp = j + 1;
                    for i = ir
                        if opts.skip_source_update && i==si && j==sj && k==sk
                            continue;
                        end
                        ip = i + 1;

                        % neighbor mins in each axis (upwind) using padded array
                        ax = min(tau_pad(ip-1, jp,   kp), tau_pad(ip+1, jp,   kp)); % x-axis neighbors
                        by = min(tau_pad(ip,   jp-1, kp), tau_pad(ip,   jp+1, kp)); % y-axis neighbors
                        cz = min(tau_pad(ip,   jp,   kp-1), tau_pad(ip, jp, kp+1)); % z-axis neighbors

                        % local slowness
                        ss = s(i,j,k);

                        % -------------------------------------------------
                        % fast local update WITHOUT sort() (3 compare-swap)
                        % We must sort (value, weight, h) together.
                        % Start with three pairs:
                        v1 = ax; w1 = invdx2; h1 = dx;
                        v2 = by; w2 = invdy2; h2 = dy;
                        v3 = cz; w3 = invdz2; h3 = dz;

                        % sort by v ascending, swapping (v,w,h) together
                        if v1 > v2
                            [v1,v2] = deal(v2,v1);
                            [w1,w2] = deal(w2,w1);
                            [h1,h2] = deal(h2,h1);
                        end
                        if v2 > v3
                            [v2,v3] = deal(v3,v2);
                            [w2,w3] = deal(w3,w2);
                            [h2,h3] = deal(h3,h2);
                        end
                        if v1 > v2
                            [v1,v2] = deal(v2,v1);
                            [w1,w2] = deal(w2,w1);
                            [h1,h2] = deal(h2,h1);
                        end

                        % 1D candidate: (t-v1)^2 / h1^2 = ss^2
                        t_new = v1 + ss*h1;
                        if t_new > v2
                            % 2D candidate with (v1,w1) and (v2,w2)
                            A = (w1 + w2);
                            B = -2*(v1*w1 + v2*w2);
                            C = (v1*v1*w1 + v2*v2*w2 - ss*ss);
                            disc = B*B - 4*A*C;
                            if disc < 0, disc = 0; end
                            t2 = (-B + sqrt(disc)) / (2*A);
                            if t2 < v2, t2 = v2; end

                            t_new = t2;
                            if t_new > v3
                                % 3D candidate with all three
                                A = (w1 + w2 + w3);
                                B = -2*(v1*w1 + v2*w2 + v3*w3);
                                C = (v1*v1*w1 + v2*v2*w2 + v3*v3*w3 - ss*ss);
                                disc = B*B - 4*A*C;
                                if disc < 0, disc = 0; end
                                t3 = (-B + sqrt(disc)) / (2*A);
                                if t3 < v3, t3 = v3; end
                                t_new = t3;
                            end
                        end
                        % -------------------------------------------------

                        if opts.check_finite && ~isfinite(t_new)
                            if info.warn_count < opts.warn_limit
                                fprintf('[WARN] tau_new nonfinite @ sweep=%d (i,j,k)=(%d,%d,%d)\n', sweep, i,j,k);
                            end
                            info.warn_count = info.warn_count + 1;
                            continue;
                        end

                        % monotone update
                        if t_new < tau(i,j,k)
                            d = tau(i,j,k) - t_new;    % positive
                            tau(i,j,k) = t_new;
                            tau_pad(ip,jp,kp) = t_new; % keep padded in sync
                            if d > maxd, maxd = d; end
                        end
                    end
                end
            end
        end

        info.iters = sweep;
        info.maxd_hist(end+1) = maxd; %#ok<AGROW>

        if opts.debug && (mod(sweep, opts.debug_every) == 0)
            fprintf('iter=%4d | max|dT|=%.3e | tau[min,max]=[%.3e, %.3e] | time=%.2fs\n', ...
                sweep, maxd, min(tau(:)), max(tau(:)), toc(t0));
        end

        if maxd < opts.tol
            info.converged = true;
            if opts.debug
                fprintf('[Eikonal_3d_1st_] Converged at iter=%d, max|dT|=%.3e\n', sweep, maxd);
                fprintf('[Eikonal_3d_1st_] Done. tau[min,max]=[%.3e, %.3e]\n', min(tau(:)), max(tau(:)));
            end
            return;
        end
    end

    if opts.debug
        fprintf('[Eikonal_3d_1st_] Reached max_sweeps=%d\n', opts.max_sweeps);
        fprintf('[Eikonal_3d_1st_] Done. tau[min,max]=[%.3e, %.3e]\n', min(tau(:)), max(tau(:)));
    end
end

% -------- helper --------
function opts = set_default(opts, name, value)
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = value;
    end
end