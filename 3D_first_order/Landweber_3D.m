%% Landweber-type Iteration for 3D Eikonal Inversion (1st-order solver)
clc; clear; close all;
tic; format long;

%% ========== Experiment Configuration ==========
regularization_type = 'L1';   % 'L1' | 'L2' | 'TV'
tol   = 1e-3;
kmax  = 400;

beta_reg   = 1.0;   % regularization strength (your "beta")
duality_r  = 1.1;   % r>1 for L^r and J_r
mu_0 = 0.8*(1 - 1/1.05);  % step size parameter
mu_1 = 600;               % step size upper bound
N = 40;
niu = 1;            % smoothing param for poisson_fft3
squareoption = 0;   % beta solver T^2 option

use_fixed_alpha = false;
alpha_fixed = 0.1;

verbose = true;
print_every = 10;

save_figures = false;
experiment_name = '3D_test';
save_dir = fullfile(pwd,'results',sprintf('%s_%s_beta%.2f_r%.2f', ...
    experiment_name, regularization_type, beta_reg, duality_r));
if save_figures && ~exist(save_dir,'dir'); mkdir(save_dir); end

%% ========== Grid (ndgrid) ==========
x = linspace(-1, 1, N);
y = linspace(-1, 1, N);
z = linspace(-1, 1, N);
[X, Y, Z] = ndgrid(x, y, z);

I=N; J=N; K=N;
dx = (x(end)-x(1))/(I-1);
dy = (y(end)-y(1))/(J-1);
dz = (z(end)-z(1))/(K-1);
h  = dx;  % assume dx=dy=dz

%% ========== Exact Model ==========
% c_exact = 3 ...
%     - 0.5 * exp(-4*( X.^2 + (Y+0.5).^2 + Z.^2 )) ...
%     - 1.0 * exp(-4*( X.^2 + (Y-0.25).^2 + Z.^2 ));
% ===== Piecewise-constant 3D velocity model with positive lower bound =====
c_min = 0.2;     % strict positive lower bound (you can change)
c_bg  = 3.0;     % background velocity

c_exact = c_bg * ones(size(X));   % start from constant background

% --- Inclusion 1: sphere (higher velocity) ---
r1   = 0.35;
ctr1 = [0.0,  0.25, 0.0];         % (x0,y0,z0)
mask1 = (X-ctr1(1)).^2 + (Y-ctr1(2)).^2 + (Z-ctr1(3)).^2 <= r1^2;
c_exact(mask1) = 3.6;

% --- Inclusion 2: rectangular box (lower velocity, but still >= c_min) ---
box = (X >= -0.70 & X <= -0.25) & ...
      (Y >= -0.20 & Y <=  0.30) & ...
      (Z >=  0.10 & Z <=  0.55);
c_exact(box) = 0.9;               % keep above c_min

% --- Inclusion 3: a slab/layer (moderately higher velocity) ---
slab = (Z >= -0.60 & Z <= -0.45); % thin layer
c_exact(slab) = 3.3;

% --- enforce strict positivity (safety clamp) ---
c_exact = max(c_exact, c_min);
% positivity / background
backCond = c_bg;

%% ========== Source Points Configuration ==========
% sourceset{p} = [xs, ys, zs] in physical coordinates.
% Use 2 planes z = ±0.75 (snapped to nearest grid nodes), each with a 3×3 grid in (x,y).
% Total sources: 18.

use_3x3 = true;

if use_3x3
    idx_xy = round(linspace(2, N-1, 3));     % e.g., [2, mid, N-1]
    xs_list = x(idx_xy);
    ys_list = y(idx_xy);

    zplanes_target = [-0.75, 0.75];
    zidx = round((zplanes_target - z(1)) / h) + 1;
    zidx = max(1, min(N, zidx));
    zplanes = z(zidx);

    [XX, YY, ZZ] = ndgrid(xs_list, ys_list, zplanes);   % 3×3×2
    S = [XX(:), YY(:), ZZ(:)];                          % 18×3
    sourceset = mat2cell(S, ones(size(S,1),1), 3);
else
    idx_xy = round(linspace(2, N-1, 7));
    xs_list = x(idx_xy);
    ys_list = y(idx_xy);

    zplanes_target = [-0.75, 0.75];
    zidx = round((zplanes_target - z(1)) / h) + 1;
    zidx = max(1, min(N, zidx));
    zplanes = z(zidx);

    [XX, YY, ZZ] = ndgrid(xs_list, ys_list, zplanes);
    S = [XX(:), YY(:), ZZ(:)];
    sourceset = mat2cell(S, ones(size(S,1),1), 3);
end

fprintf('Number of sources: %d\n', numel(sourceset));

%% ========== IMPORTANT FIX 1: physical coords -> grid indices (1-based) ==========
S_all = cell2mat(sourceset);   % (P×3) physical coords

ix = arrayfun(@(v) find(abs(x - v) == min(abs(x - v)), 1, 'first'), S_all(:,1));
iy = arrayfun(@(v) find(abs(y - v) == min(abs(y - v)), 1, 'first'), S_all(:,2));
iz = arrayfun(@(v) find(abs(z - v) == min(abs(z - v)), 1, 'first'), S_all(:,3));

S_idx = [ix, iy, iz];          % (P×3) integer indices
sourceset_idx = mat2cell(S_idx, ones(size(S_idx,1),1), 3);

% ---- plot sources (physical) for sanity ----
figure;
scatter3(S_all(:,1), S_all(:,2), S_all(:,3), 60, 'filled');
grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Source locations (physical coords)');

%% ========== Initial Guess c0 ==========
% c0 = poisson3d_fd_fullbc(zeros(I-2,J-2,K-2), c_exact, h, niu); % T.B.D
% interior RHS f (nx,ny,nz)
f0 = zeros(I-2, J-2, K-2);

% build full-grid Dirichlet bc with one-layer ghost shell
bc_full = zeros(I, J, K);
bc_full(1,:,:)   = c_exact(1,:,:);
bc_full(end,:,:) = c_exact(end,:,:);
bc_full(:,1,:)   = c_exact(:,1,:);
bc_full(:,end,:) = c_exact(:,end,:);
bc_full(:,:,1)   = c_exact(:,:,1);
bc_full(:,:,end) = c_exact(:,:,end);

% solve (-niu*Δu + u = f) with Dirichlet bc
c0 = poisson3d_fd_fullbc_xyz(f0, bc_full, dx, dy, dz, niu);
%% ========== Initialize xi then c via regularization (3D) ==========
switch regularization_type
    case 'L2'
        xi = c0;

    case 'L1'
        q0 = sign(c0 - backCond);
        xi = c0 + beta_reg*q0;

    case 'TV'
        if beta_reg <= 0
            xi = c0;
        else
            lbd = 1/beta_reg;
            NIT = 200; GapTol = 1e-6;
            w1 = zeros(size(c0)); w2 = w1; w3 = w1;
            [~, w1, w2, w3] = TV_PDHD_3D(w1,w2,w3,c0,lbd,NIT,GapTol);

            div_w = cat(1, w1(1,:,:) , w1(2:end,:,:) - w1(1:end-1,:,:)) ...
                  + cat(2, w2(:,1,:) , w2(:,2:end,:) - w2(:,1:end-1,:)) ...
                  + cat(3, w3(:,:,1) , w3(:,:,2:end) - w3(:,:,1:end-1));

            xi = c0 + beta_reg * (-div_w);
        end

    otherwise
        error('Unknown regularization type.');
end

%% ========== Separate opts for Eikonal vs Regularization ==========
opts_eik = struct();
opts_eik.xlim = [-1, 1];
opts_eik.ylim = [-1, 1];
opts_eik.zlim = [-1, 1];
opts_eik.verbose = false;   % parfor 下强烈建议关输出（你的 solver 里可自行支持）

opts_reg = struct();        % 传给 apply_regularization3D（如不需要可空）

c = apply_regularization3D(xi, regularization_type, beta_reg, backCond, c_min, opts_reg);

%% ========== Precompute Observation Data T_star ==========
T_star = cell(numel(sourceset_idx), 1);

parfor p = 1:numel(sourceset_idx)
    T_star{p} = Eikonal_3d_1st_(c_exact, sourceset_idx{p}, dx, dy, dz, opts_eik);
end

%% ========== Main Iteration ==========
energy    = zeros(kmax,1);
resn_set  = zeros(kmax,1);
alpha_set = zeros(kmax,1);
cerror    = zeros(kmax,1);
dual_norm_set = zeros(kmax,1);

fprintf('\n========== Starting Iteration ==========\n');
for k = 1:kmax

    Ep_list     = zeros(numel(sourceset_idx),1);
    resPow_list = zeros(numel(sourceset_idx),1);
    cstar_list  = cell(numel(sourceset_idx),1);

    parfor p_num = 1:numel(sourceset_idx)

        % =======================
        % forward 
        % =======================
        % T = Eikonal_3d_1st_(c, sourceset_idx{p_num}, dx, dy, dz, opts_eik);
        T = Eikonal_3d_1st_A(c, sourceset_idx{p_num}, dx, dy, dz, opts_eik);

        % energy (your L2 boundary energy)
        Ep_list(p_num) = EnergyFun3D(T, T_star{p_num}, dx, dy, dz);

        % L^r residual on boundary for this source
        rp = LrNormBoundary3D(T, T_star{p_num}, dx, dy, dz, duality_r);
        resPow_list(p_num) = rp^duality_r;

        % J_r(res) (Banach duality mapping)
        res = T_star{p_num} - T;
        J_res = sign(res) .* abs(res).^(duality_r-1);

        % solve beta with rhs = J_r(residual)
        % betap = beta_solver_3d_2(T, J_res, dx, dy, dz, squareoption);
        % betap = beta_solver_3d_fast(T, J_res, dx, dy, dz, squareoption);
         betap = beta2_solver_3d(T, J_res, dx, dy, dz)

        % chain rule part
        ctemp = -betap ./ (c.^3);

        % smoothing (interior only)
        ctemp2 = zeros(size(ctemp));
        ctemp2(2:end-1,2:end-1,2:end-1) = poisson_fft3( ...
            ctemp(2:end-1,2:end-1,2:end-1), dx, dy, dz, niu);

        cstar_list{p_num} = ctemp2;
    end

    energy_p = sum(Ep_list);
    res_Lr   = (sum(resPow_list))^(1/duality_r);

    % accumulate cstar
    cstar = zeros(I,J,K);
    for p_num = 1:numel(sourceset_idx)
        cstar = cstar + cstar_list{p_num};
    end

    % record
    energy(k)   = energy_p;
    resn_set(k) = res_Lr;
    cerror(k)   = sum(abs(c(:)-c_exact(:))) * dx*dy*dz;

    if energy_p < tol
        fprintf('Converged at iter %d: Energy=%.3e\n', k, energy_p);
        break;
    end

    % ===== Compute dual norm =====
    g = cstar;
    norm_dual_sq = sum(g(:).^2) * dx * dy * dz;  % ||g||^2_{L2}
    dual_norm_set(k) = norm_dual_sq;

    % step size
    if use_fixed_alpha
        alpha = alpha_fixed;
    else
        alpha = min(mu_0 * res_Lr^(2*(duality_r - 1)) / max(norm_dual_sq, 1e-12), mu_1) ...
                * res_Lr^(2-duality_r);
    end
    alpha_set(k) = alpha;

    % update xi then apply prox
    xi = xi + alpha * cstar;
    c  = apply_regularization3D(xi, regularization_type, beta_reg, backCond, c_min, opts_reg);

    if verbose && (mod(k,print_every)==0 || k==1)
        fprintf('Iter %4d | Energy=%.3e | Res_Lr=%.3e | Err(L1)=%.3e | alpha=%.3e\n', ...
            k, energy_p, res_Lr, cerror(k), alpha);
    end
end

total_iterations = k;
elapsed_time = toc;

% trim history
energy    = energy(1:total_iterations);
resn_set  = resn_set(1:total_iterations);
alpha_set = alpha_set(1:total_iterations);
cerror    = cerror(1:total_iterations);
dual_norm_set = dual_norm_set(1:total_iterations);

fprintf('\n========== Summary ==========\n');
fprintf('Total iterations: %d\n', total_iterations);
fprintf('Elapsed time: %.2f s\n', elapsed_time);

%% ========== Visualization ==========
mid = ceil(N/2);

cmin_sol = min([c_exact(:); c0(:); c(:)]);
cmax_sol = max([c_exact(:); c0(:); c(:)])/3;

fig1 = figure('Position',[80 80 1400 650]);
sgtitle(sprintf('3D Inversion (%s, beta=%.2f, r=%.2f)', ...
    regularization_type, beta_reg, duality_r), 'FontWeight','bold');

subplot(2,3,1);
imagesc(x,y,squeeze(c_exact(:,:,mid))'); axis xy equal tight; colorbar;
caxis([cmin_sol cmax_sol]); title('c_{exact} (z mid)'); xlabel('x'); ylabel('y');

subplot(2,3,2);
imagesc(x,y,squeeze(c0(:,:,mid))'); axis xy equal tight; colorbar;
caxis([cmin_sol cmax_sol]); title('c_0 (z mid)'); xlabel('x'); ylabel('y');

subplot(2,3,3);
imagesc(x,y,squeeze(c(:,:,mid))'); axis xy equal tight; colorbar;
caxis([cmin_sol cmax_sol]); title('c (z mid)'); xlabel('x'); ylabel('y');

subplot(2,3,4);
imagesc(x, y, squeeze(abs(c(:,:,mid) - c_exact(:,:,mid)))');
axis xy equal tight; colorbar;
title('|c-c_{exact}| (z mid)'); xlabel('x'); ylabel('y');

subplot(2,3,5); hold on;
plot(x, squeeze(c_exact(:,mid,mid)),'LineWidth',1.8);
plot(x, squeeze(c0(:,mid,mid)),'--','LineWidth',1.5);
plot(x, squeeze(c(:,mid,mid)),':','LineWidth',1.8);
grid on; xlabel('x'); ylabel('c');
title('Center line (y=mid,z=mid)');
legend({'c_{exact}','c_0','c'},'Location','best'); hold off;

subplot(2,3,6); hold on;
semilogy(energy,'LineWidth',1.5);
semilogy(resn_set,'LineWidth',1.5);
semilogy(cerror,'LineWidth',1.5);
grid on; xlabel('iter'); title('Convergence (log)');
legend({'Energy','Res L^r','Err(L1)'},'Location','best'); hold off;

if save_figures
    saveas(fig1, fullfile(save_dir,'summary.png'));
    saveas(fig1, fullfile(save_dir,'summary.fig'));
    fprintf('Saved: summary.png\n');
end

%% ========== Figure 2: 3D volume with removed box (slice) ==========
field_name3D = 'c';   % 'c_exact' | 'c0' | 'c'
switch field_name3D
    case 'c_exact', C = c_exact;
    case 'c0',      C = c0;
    case 'c',       C = c;
end

xslice = 0; yslice = 0; zslice = 0;

fig2 = figure('Position',[80 80 1600 520]);
sgtitle(sprintf('3D slice (%s): removed region shown as NaN', field_name3D), ...
    'FontWeight','bold');

% -------------------------
% Case A: remove 1/2 volume (x > 0)
% -------------------------
Cmask = C;
Cmask(x > 0, :, :) = NaN;

subplot(1,3,1);
V = permute(Cmask, [2 1 3]);                 % ndgrid -> meshgrid convention
h = slice(x, y, z, V, xslice, yslice, zslice);
set(h,'EdgeColor','none'); axis equal tight; view(3);
xlabel('x'); ylabel('y'); zlabel('z'); colorbar;
caxis([cmin_sol cmax_sol]);
title('Remove 1/2: x>0');

% -------------------------
% Case B: remove 1/4 volume (x > 0 & y > 0)
% -------------------------
Cmask = C;
Cmask(x > 0, y > 0, :) = NaN;

subplot(1,3,2);
V = permute(Cmask, [2 1 3]);
h = slice(x, y, z, V, xslice, yslice, zslice);
set(h,'EdgeColor','none'); axis equal tight; view(3);
xlabel('x'); ylabel('y'); zlabel('z'); colorbar;
caxis([cmin_sol cmax_sol]);
title('Remove 1/4: x>0,y>0');

% -------------------------
% Case C: remove 3/4 volume (keep only x<=0 & y<=0)
% -------------------------
Cmask = NaN(size(C));
Cmask(x <= 0, y <= 0, :) = C(x <= 0, y <= 0, :);

subplot(1,3,3);
V = permute(Cmask, [2 1 3]);
h = slice(x, y, z, V, xslice, yslice, zslice);
set(h,'EdgeColor','none'); axis equal tight; view(3);
xlabel('x'); ylabel('y'); zlabel('z'); colorbar;
caxis([cmin_sol cmax_sol]);
title('Remove 3/4: keep x<=0,y<=0');

if save_figures
    saveas(fig2, fullfile(save_dir, sprintf('3D_removed_%s.png', field_name3D)));
    saveas(fig2, fullfile(save_dir, sprintf('3D_removed_%s.fig', field_name3D)));
end