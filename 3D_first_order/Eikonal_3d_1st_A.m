function [tau, info] = Eikonal_3d_1st_A(c, src_idx, dx, dy, dz, opts)
% EIKONAL_3D_1ST_FASTA
% Same method as Eikonal_3d_1st_ (1st-order upwind + fast sweeping),
% but faster: padding + no sort() + precompute + fewer function calls.

    if nargin < 6, opts = struct(); end
    if ~isfield(opts,'max_sweeps'), opts.max_sweeps = 200; end
    if ~isfield(opts,'tol'),        opts.tol = 1e-6;       end
    if ~isfield(opts,'large'),      opts.large = 1e9;      end
    if ~isfield(opts,'debug'),      opts.debug = false;    end
    if ~isfield(opts,'debug_every'),opts.debug_every = 1;  end
    if ~isfield(opts,'skip_source_update'), opts.skip_source_update = true; end

    [I,J,K] = size(c);
    si = src_idx(1); sj = src_idx(2); sk = src_idx(3);
    if any(src_idx < 1) || si>I || sj>J || sk>K
        error('src_idx out of bounds.');
    end
    if dx<=0 || dy<=0 || dz<=0
        error('dx,dy,dz must be positive.');
    end
    if any(~isfinite(c(:))) || any(c(:) <= 0)
        c = max(c, eps);
    end

    % Precompute slowness
    s = 1 ./ c;

    % Precompute constants
    invdx2 = 1/(dx*dx); invdy2 = 1/(dy*dy); invdz2 = 1/(dz*dz);

    % If almost uniform grid, we can use cheaper branch
    uniformGrid = (abs(dx-dy) + abs(dx-dz) + abs(dy-dz)) < 1e-14;

    % Info
    info = struct();
    info.converged = false;
    info.iters = 0;
    info.maxd_hist = zeros(opts.max_sweeps+1,1);

    % Padding (ghost layer) to remove bounds checks
    L = opts.large;
    tauP = L * ones(I+2, J+2, K+2, 'like', c);  % padded tau
    tauP(2:I+1, 2:J+1, 2:K+1) = L;
    tauP(si+1, sj+1, sk+1) = 0;

    % Precompute sweep index orders (in padded coordinates)
    irs = { 2:I+1,   I+1:-1:2 };
    jrs = { 2:J+1,   J+1:-1:2 };
    krs = { 2:K+1,   K+1:-1:2 };
    % 8 directions (choice of forward/backward in each axis)
    dirs = { ...
        irs{1}, jrs{1}, krs{1}; ...
        irs{1}, jrs{1}, krs{2}; ...
        irs{1}, jrs{2}, krs{1}; ...
        irs{1}, jrs{2}, krs{2}; ...
        irs{2}, jrs{1}, krs{1}; ...
        irs{2}, jrs{1}, krs{2}; ...
        irs{2}, jrs{2}, krs{1}; ...
        irs{2}, jrs{2}, krs{2}  ...
    };

    % Source in padded coordinates
    sip = si+1; sjp = sj+1; skp = sk+1;

    % Sweeps
    for sweep = 0:opts.max_sweeps
        t0 = tic;
        maxd = 0;

        for dd = 1:8
            ir = dirs{dd,1};
            jr = dirs{dd,2};
            kr = dirs{dd,3};

            for kp = kr
                for jp = jr
                    for ip = ir
                        if opts.skip_source_update && ip==sip && jp==sjp && kp==skp
                            continue;
                        end

                        % neighbor mins (upwind)
                        ax = min(tauP(ip-1,jp,kp), tauP(ip+1,jp,kp));
                        by = min(tauP(ip,jp-1,kp), tauP(ip,jp+1,kp));
                        cz = min(tauP(ip,jp,kp-1), tauP(ip,jp,kp+1));

                        % local slowness (note: s is unpadded)
                        sp = s(ip-1, jp-1, kp-1);

                        told = tauP(ip,jp,kp);

                        % -------- local update (inlined, no function calls) --------
                        if uniformGrid
                            % only sort a,b,c (spacing same)
                            a = ax; b = by; c3 = cz;
                            % sort network for 3 numbers (cheap, no sort())
                            if a > b, tmp=a; a=b; b=tmp; end
                            if b > c3, tmp=b; b=c3; c3=tmp; end
                            if a > b, tmp=a; a=b; b=tmp; end

                            % 1D
                            tnew = a + sp*dx;
                            if tnew > b
                                % 2D
                                A = invdx2 + invdy2; % same as 2*invdx2 actually, but keep form
                                B = -2*(a*invdx2 + b*invdy2);
                                C = (a*a)*invdx2 + (b*b)*invdy2 - sp*sp;
                                disc = B*B - 4*A*C;
                                if disc < 0, disc = 0; end
                                t2 = (-B + sqrt(disc)) / (2*A);
                                if t2 < b, t2 = b; end
                                tnew = t2;

                                if tnew > c3
                                    % 3D
                                    A = invdx2 + invdy2 + invdz2;
                                    B = -2*(a*invdx2 + b*invdy2 + c3*invdz2);
                                    C = (a*a)*invdx2 + (b*b)*invdy2 + (c3*c3)*invdz2 - sp*sp;
                                    disc = B*B - 4*A*C;
                                    if disc < 0, disc = 0; end
                                    t3 = (-B + sqrt(disc)) / (2*A);
                                    m = c3; % max(a,b,c3)=c3 after sorting
                                    if t3 < m, t3 = m; end
                                    tnew = t3;
                                end
                            end

                        else
                            % general dx,dy,dz: sort (tau, invh2) pairs cheaply
                            t1 = ax; w1 = invdx2; h1 = dx;
                            t2 = by; w2 = invdy2; h2 = dy;
                            t3 = cz; w3 = invdz2; h3 = dz;

                            if t1 > t2, tmp=t1; t1=t2; t2=tmp; tmp=w1; w1=w2; w2=tmp; tmp=h1; h1=h2; h2=tmp; end
                            if t2 > t3, tmp=t2; t2=t3; t3=tmp; tmp=w2; w2=w3; w3=tmp; tmp=h2; h2=h3; h3=tmp; end
                            if t1 > t2, tmp=t1; t1=t2; t2=tmp; tmp=w1; w1=w2; w2=tmp; tmp=h1; h1=h2; h2=tmp; end

                            % 1D with matching spacing
                            tnew = t1 + sp*h1;
                            if tnew > t2
                                % 2D
                                A = w1 + w2;
                                B = -2*(t1*w1 + t2*w2);
                                C = (t1*t1)*w1 + (t2*t2)*w2 - sp*sp;
                                disc = B*B - 4*A*C; if disc < 0, disc = 0; end
                                tnew = (-B + sqrt(disc)) / (2*A);
                                if tnew < t2, tnew = t2; end

                                if tnew > t3
                                    % 3D
                                    A = w1 + w2 + w3;
                                    B = -2*(t1*w1 + t2*w2 + t3*w3);
                                    C = (t1*t1)*w1 + (t2*t2)*w2 + (t3*t3)*w3 - sp*sp;
                                    disc = B*B - 4*A*C; if disc < 0, disc = 0; end
                                    tnew = (-B + sqrt(disc)) / (2*A);
                                    if tnew < t3, tnew = t3; end
                                end
                            end
                        end
                        % ---------------------------------------------------------

                        if tnew < told
                            tauP(ip,jp,kp) = tnew;
                            d = told - tnew;
                            if d > maxd, maxd = d; end
                        end
                    end
                end
            end
        end

        info.iters = sweep;
        info.maxd_hist(sweep+1) = maxd;

        if opts.debug && (mod(sweep, opts.debug_every)==0)
            tauInterior = tauP(2:I+1,2:J+1,2:K+1);
            fprintf('iter=%4d | max|dT|=%.3e | tau[min,max]=[%.3e, %.3e] | time=%.2fs\n', ...
                sweep, maxd, min(tauInterior(:)), max(tauInterior(:)), toc(t0));
        end

        if maxd < opts.tol
            info.converged = true;
            break;
        end
    end

    % unpad
    tau = tauP(2:I+1,2:J+1,2:K+1);

    % trim hist
    info.maxd_hist = info.maxd_hist(1:info.iters+1);
end