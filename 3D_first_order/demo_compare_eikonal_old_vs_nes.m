%% demo_compare_eikonal_old_vs_new.m
clc; clear; close all;

fprintf('=== Compare Eikonal_3d_1st (old) vs Eikonal_3d_1st_ (new) ===\n');

%% ----------------------------
% Grid
%% ----------------------------
N = 51;
computex = [-1, 1];
computey = [-1, 1];
computez = [-1, 1];

h  = (computex(2)-computex(1))/(N-1);
dx = h; dy = h; dz = h;

x = linspace(computex(1), computex(2), N);
y = linspace(computey(1), computey(2), N);
z = linspace(computez(1), computez(2), N);

% ndgrid gives arrays size (N,N,N) with i->x, j->y, k->z (matches your new solver)
[X,Y,Z] = ndgrid(x,y,z);

%% ----------------------------
% Velocity model c(x,y,z) (positive)
%% ----------------------------
c_min = 0.5;
c_bg  = 3.0;
c = c_bg*ones(size(X));

% a sphere inclusion
mask1 = (X-0.1).^2 + (Y+0.2).^2 + (Z-0.05).^2 <= 0.35^2;
c(mask1) = 3.6;

% a low-velocity box (still positive)
mask2 = (X >= -0.7 & X <= -0.25) & (Y >= -0.2 & Y <= 0.3) & (Z >= 0.1 & Z <= 0.55);
c(mask2) = 0.9;

c = max(c, c_min);

% old solver needs n2 = 1/c^2 (slowness^2)
n2 = 1./(c.^2);

%% ----------------------------
% Source (physical)  (NOT exactly on grid to avoid 0/0 in old code)
%% ----------------------------
source = [0.13, -0.07, 0.21];  % (x,y,z) physical

% old code rounds to indices:
src_i = round((source(1) - computex(1))/h) + 1;
src_j = round((source(2) - computey(1))/h) + 1;
src_k = round((source(3) - computez(1))/h) + 1;

src_i = max(1, min(N, src_i));
src_j = max(1, min(N, src_j));
src_k = max(1, min(N, src_k));

src_idx = [src_i, src_j, src_k];

fprintf('Grid: N=%d, h=%.5f\n', N, h);
fprintf('source(phys) = [%.3f, %.3f, %.3f]\n', source(1), source(2), source(3));
fprintf('source(idx ) = [%d, %d, %d]\n', src_idx(1), src_idx(2), src_idx(3));

%% ----------------------------
% Options for new solver
%% ----------------------------
opts = struct();
opts.max_sweeps = 50;
opts.tol = 1e-5;
opts.large = 1e9;
opts.debug = false;
opts.skip_source_update = true;

%% ----------------------------
% Warm-up (JIT)
%% ----------------------------
tau_old = Eikonal_3d_1st(computex, computey, computez, source, h, n2); %#ok<NASGU>
tau_new = Eikonal_3d_1st_(c, src_idx, dx, dy, dz, opts); %#ok<NASGU>
% fprintf('new solver sweeps = %d\n', info.iters);
%% ----------------------------
% Timing (avg of nrep)
%% ----------------------------
nrep = 3;
t_old = zeros(nrep,1);
t_new = zeros(nrep,1);

for k = 1:nrep
    tic;
    tau_old = Eikonal_3d_1st(computex, computey, computez, source, h, n2);
    t_old(k) = toc;

    tic;
    tau_new = Eikonal_3d_1st_(c, src_idx, dx, dy, dz, opts);
    t_new(k) = toc;
end

t1 = mean(t_old);
t2 = mean(t_new);

fprintf('\n=== Timing ===\n');
fprintf('time(old Eikonal_3d_1st)   = %.4f s (avg of %d)\n', t1, nrep);
fprintf('time(new Eikonal_3d_1st_)  = %.4f s (avg of %d)\n', t2, nrep);
fprintf('speedup = %.2fx\n', t1/max(t2,eps));

%% ----------------------------
% Accuracy metrics
%% ----------------------------
diff = tau_new - tau_old;
maxdiff = max(abs(diff(:)));
relL2   = norm(diff(:)) / max(norm(tau_old(:)), eps);

fprintf('\n=== Difference ===\n');
fprintf('max|tau_new - tau_old| = %.3e\n', maxdiff);
fprintf('relL2(tau_new - tau_old) = %.3e\n', relL2);

%% ----------------------------
% Visual check: mid-z slice
%% ----------------------------
mid = ceil(N/2);

figure('Position',[80 80 1400 450]);
sgtitle('Eikonal comparison (mid-z slice)','FontWeight','bold');

subplot(1,3,1);
imagesc(x,y, squeeze(tau_old(:,:,mid))'); axis xy equal tight; colorbar;
title('tau\_old (mid z)'); xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(x,y, squeeze(tau_new(:,:,mid))'); axis xy equal tight; colorbar;
title('tau\_new (mid z)'); xlabel('x'); ylabel('y');

subplot(1,3,3);
imagesc(x,y, squeeze(abs(diff(:,:,mid)))'); axis xy equal tight; colorbar;
title('|tau\_new - tau\_old| (mid z)'); xlabel('x'); ylabel('y');

fprintf('\nDone.\n');