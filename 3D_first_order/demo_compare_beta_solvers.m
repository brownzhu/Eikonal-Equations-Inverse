%% demo_compare_beta_solvers.m
clc; clear; close all;

fprintf('=== Compare beta_solver_3d_2 (original) vs beta_solver_3d_fast (new) ===\n');

%% -------------------------
% 0) Setup grid + synthetic T
%% -------------------------
N = 51;                         % grid size (try 25/51/75)
x = linspace(-1,1,N);
y = linspace(-1,1,N);
z = linspace(-1,1,N);
dx = x(2)-x(1); dy = y(2)-y(1); dz = z(2)-z(1);

[X,Y,Z] = ndgrid(x,y,z);

% constant velocity -> travel time like distance/c
c0 = 3.0;
src = [0.2, -0.1, 0.3];         % physical source position (not necessarily on grid)
T = sqrt((X-src(1)).^2 + (Y-src(2)).^2 + (Z-src(3)).^2) / c0;

squareoption = 1;               % match your main code: 1 means use T^2 in coeff

%% -------------------------
% 1) Build res: nonzero ONLY on boundary
%% -------------------------
rng(0);
res = zeros(size(T));

% put random data on 6 faces
res(1,:,:)   = randn(1,N,N);
res(end,:,:) = randn(1,N,N);
res(:,1,:)   = randn(N,1,N);
res(:,end,:) = randn(N,1,N);
res(:,:,1)   = randn(N,N,1);
res(:,:,end) = randn(N,N,1);

% (optional) keep corners/edges consistent? not necessary for comparison.

%% -------------------------
% 2) Warm-up (avoid first-call overhead)
%% -------------------------
beta_solver_3d_2(T, res, dx, dy, dz, squareoption);
beta_solver_3d_fast(T, res, dx, dy, dz, squareoption);

%% -------------------------
% 3) Timing (average over repeats)
%% -------------------------
nRepeat = 3;

t_old = 0;
for r = 1:nRepeat
    tic;
    beta_old = beta_solver_3d_2(T, res, dx, dy, dz, squareoption);
    t_old = t_old + toc;
end
t_old = t_old/nRepeat;

t_new = 0;
for r = 1:nRepeat
    tic;
    beta_new = beta_solver_3d_fast(T, res, dx, dy, dz, squareoption);
    t_new = t_new + toc;
end
t_new = t_new/nRepeat;

%% -------------------------
% 4) Accuracy comparison
%% -------------------------
diff = beta_new - beta_old;
maxAbs = max(abs(diff(:)));
relL2  = norm(diff(:)) / max(norm(beta_old(:)), 1e-15);

fprintf('Grid: N=%d, dx=dy=dz=%.5f, squareoption=%d\n', N, dx, squareoption);
fprintf('time(original beta_solver_3d_2) = %.4f s (avg of %d)\n', t_old, nRepeat);
fprintf('time(new beta_solver_3d_fast)   = %.4f s (avg of %d)\n', t_new, nRepeat);
fprintf('speedup = %.2fx\n', t_old / t_new);
fprintf('max|beta_new-beta_old| = %.3e\n', maxAbs);
fprintf('relL2(beta_new-beta_old) = %.3e\n', relL2);

%% -------------------------
% 5) Boundary condition residual check:
%    (n·∇T)*beta = res on each face
%    Using same one-sided FD as in your solver.
%% -------------------------
inv_dx = 1/dx; inv_dy = 1/dy; inv_dz = 1/dz;

% define n·∇T on faces (same as your beta_solver_3d_fast)
gL  = -(T(:,2,:) - T(:,1,:))   * inv_dy;   % j=1
gR  =  (T(:,end,:) - T(:,end-1,:)) * inv_dy; % j=J
gT  = -(T(2,:,:) - T(1,:,:))   * inv_dx;   % i=1
gB  =  (T(end,:,:) - T(end-1,:,:)) * inv_dx; % i=I
gF  = -(T(:,:,2) - T(:,:,1))   * inv_dz;   % k=1
gBk =  (T(:,:,end) - T(:,:,end-1)) * inv_dz; % k=K

% compute BC residuals for old/new
bc_inf_old = max([
    max(abs(gL .* beta_old(:,1,:)   - res(:,1,:)),   [], 'all')
    max(abs(gR .* beta_old(:,end,:) - res(:,end,:)), [], 'all')
    max(abs(gT .* beta_old(1,:,:)   - res(1,:,:)),   [], 'all')
    max(abs(gB .* beta_old(end,:,:) - res(end,:,:)), [], 'all')
    max(abs(gF .* beta_old(:,:,1)   - res(:,:,1)),   [], 'all')
    max(abs(gBk.* beta_old(:,:,end) - res(:,:,end)), [], 'all')
]);

bc_inf_new = max([
    max(abs(gL .* beta_new(:,1,:)   - res(:,1,:)),   [], 'all')
    max(abs(gR .* beta_new(:,end,:) - res(:,end,:)), [], 'all')
    max(abs(gT .* beta_new(1,:,:)   - res(1,:,:)),   [], 'all')
    max(abs(gB .* beta_new(end,:,:) - res(end,:,:)), [], 'all')
    max(abs(gF .* beta_new(:,:,1)   - res(:,:,1)),   [], 'all')
    max(abs(gBk.* beta_new(:,:,end) - res(:,:,end)), [], 'all')
]);

fprintf('\n=== BC check (max face residual) ===\n');
fprintf('|| (n·∇T)*beta_old - res ||_inf = %.3e\n', bc_inf_old);
fprintf('|| (n·∇T)*beta_new - res ||_inf = %.3e\n', bc_inf_new);

%% -------------------------
% 6) Optional: visualize a mid slice of diff
%% -------------------------
mid = ceil(N/2);
figure('Position',[100 100 1200 350]);

subplot(1,3,1);
imagesc(x,y,squeeze(beta_old(:,:,mid))'); axis xy equal tight; colorbar;
title('beta\_old (mid z)'); xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(x,y,squeeze(beta_new(:,:,mid))'); axis xy equal tight; colorbar;
title('beta\_new (mid z)'); xlabel('x'); ylabel('y');

subplot(1,3,3);
imagesc(x,y,squeeze(diff(:,:,mid))'); axis xy equal tight; colorbar;
title('diff = new-old (mid z)'); xlabel('x'); ylabel('y');

fprintf('\nDone.\n');