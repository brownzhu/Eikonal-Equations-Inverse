%% demo_compare_Eikonal_vs_truth.m
clc; clear; close all;

%% ========== grid / domain ==========
computex = [-1, 1];
computey = [-1, 1];
computez = [-1, 1];

h  = 0.04;              % dx=dy=dz
dx = h; dy = h; dz = h;

nx = round((computex(2)-computex(1))/h)+1;
ny = round((computey(2)-computey(1))/h)+1;
nz = round((computez(2)-computez(1))/h)+1;

x = computex(1):h:computex(2);
y = computey(1):h:computey(2);
z = computez(1):h:computez(2);

% sanity
assert(numel(x)==nx && numel(y)==ny && numel(z)==nz, 'grid size mismatch');

[X,Y,Z] = ndgrid(x,y,z);   % IMPORTANT: (i,j,k) aligns with your arrays

%% ========== source ==========
src_phys = [0.130, -0.070, 0.210];  % physical location
src_idx  = round((src_phys - [computex(1),computey(1),computez(1)])/h) + 1;
src_idx  = max([1,1,1], min([nx,ny,nz], src_idx)); % clamp

fprintf('Grid: (nx,ny,nz)=(%d,%d,%d), h=%.5f\n', nx, ny, nz, h);
fprintf('src_phys=[%.3f %.3f %.3f], src_idx=[%d %d %d]\n', ...
    src_phys(1),src_phys(2),src_phys(3), src_idx(1),src_idx(2),src_idx(3));

%% ========== choose an EXACT solution case ==========
case_name = 'radial';   % 'constant' | 'radial'

r = sqrt( (X-src_phys(1)).^2 + (Y-src_phys(2)).^2 + (Z-src_phys(3)).^2 );

switch lower(case_name)
    case 'constant'
        c0 = 3.0;
        c_true   = c0 * ones(nx,ny,nz);
        tau_true = r / c0;

    case 'radial'
        % c(r) = c0*(1+alpha*r)  => tau(r) = (1/(c0*alpha))*log(1+alpha*r)
        c0    = 2.5;
        alpha = 0.8;                 % must be >0
        c_true   = c0 * (1 + alpha*r);
        tau_true = (1/(c0*alpha)) * log(1 + alpha*r);

    otherwise
        error('Unknown case_name.');
end

% old solver uses n2 = 1/c^2 (slowness^2)
n2 = 1 ./ (c_true.^2);

%% ========== run OLD solver ==========
nrep = 3;
t_old = zeros(nrep,1);
for rr = 1:nrep
    tic;
    tau_old = Eikonal_3d_1st(computex, computey, computez, src_phys, h, n2);
    t_old(rr) = toc;
end
time_old = mean(t_old);

%% ========== run NEW solver ==========
opts = struct();
opts.max_sweeps = 200;
opts.tol        = 1e-6;
opts.large      = 1e9;
opts.debug      = false;
opts.check_finite = true;
opts.skip_source_update = true;

t_new = zeros(nrep,1);
for rr = 1:nrep
    tic;
    [tau_new, info] = Eikonal_3d_1st_A(c_true, src_idx, dx, dy, dz, opts);
    t_new(rr) = toc;
end
time_new = mean(t_new);

%% ========== compare vs truth ==========
% ignore source point in norms (optional)
mask = true(nx,ny,nz);
mask(src_idx(1),src_idx(2),src_idx(3)) = false;

err_old = tau_old - tau_true;
err_new = tau_new - tau_true;

% whole domain
max_old = max(abs(err_old(mask)));
max_new = max(abs(err_new(mask)));

relL2_old = norm(err_old(mask)) / max(norm(tau_true(mask)), 1e-14);
relL2_new = norm(err_new(mask)) / max(norm(tau_true(mask)), 1e-14);

% boundary only (since your inversion uses boundary a lot)
bd = false(nx,ny,nz);
bd(1,:,:) = true; bd(nx,:,:) = true;
bd(:,1,:) = true; bd(:,ny,:) = true;
bd(:,:,1) = true; bd(:,:,nz) = true;
bd(src_idx(1),src_idx(2),src_idx(3)) = false;

relL2_bd_old = norm(err_old(bd)) / max(norm(tau_true(bd)), 1e-14);
relL2_bd_new = norm(err_new(bd)) / max(norm(tau_true(bd)), 1e-14);

max_bd_old = max(abs(err_old(bd)));
max_bd_new = max(abs(err_new(bd)));

%% ========== report ==========
fprintf('\n=== Case: %s (exact solution known) ===\n', case_name);

fprintf('\n=== Timing (avg of %d) ===\n', nrep);
fprintf('time(old Eikonal_3d_1st)  = %.4f s\n', time_old);
fprintf('time(new Eikonal_3d_1st_) = %.4f s\n', time_new);
fprintf('speedup = %.2fx\n', time_old / time_new);

fprintf('\n=== Error vs TRUE tau (whole domain, exclude source point) ===\n');
fprintf('old: max|err| = %.3e, relL2 = %.3e\n', max_old, relL2_old);
fprintf('new: max|err| = %.3e, relL2 = %.3e\n', max_new, relL2_new);

fprintf('\n=== Error vs TRUE tau (boundary only) ===\n');
fprintf('old: max|err|_bd = %.3e, relL2_bd = %.3e\n', max_bd_old, relL2_bd_old);
fprintf('new: max|err|_bd = %.3e, relL2_bd = %.3e\n', max_bd_new, relL2_bd_new);

%% ========== quick visualization (optional) ==========
mid = round(nz/2);
figure('Position',[100 100 1400 420]);
sgtitle(sprintf('Eikonal compare to truth (%s), z-slice mid', case_name), 'FontWeight','bold');

subplot(1,3,1);
imagesc(x,y, squeeze(tau_true(:,:,mid))'); axis xy equal tight; colorbar;
title('\tau_{true}'); xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(x,y, squeeze(abs(err_old(:,:,mid)))'); axis xy equal tight; colorbar;
title('|err_{old}|'); xlabel('x'); ylabel('y');

subplot(1,3,3);
imagesc(x,y, squeeze(abs(err_new(:,:,mid)))'); axis xy equal tight; colorbar;
title('|err_{new}|'); xlabel('x'); ylabel('y');