% Landweber iteration for RECTANGULAR (non-square) domain
% Add the accelerate combination parameter.
% Tests the solver with dx ≠ dy
% Supports L1, L2, TV regularization
clc; clear; close all
tic
format long

%% ========== Experiment Configuration ==========
% Regularization type: 'L1', 'L2', 'TV'
% regularization_type = 'L1';
regularization_type = 'L2';
% regularization_type = 'TV';

% Whether to save figures
save_figures = false;

% Experiment name
experiment_name = 'rectangular_domain_Nes_';

%% ========== Rectangular Domain Configuration ==========
% Domain: [x_min, x_max] x [z_min, z_max]
% Here we test a 2:1 aspect ratio rectangle

x_min = -2; x_max = 2;   % x range: 4 units
z_min = -1; z_max = 1;   % z range: 2 units

% Grid points (different in x and z directions)
Nx = 129;    % points in x-direction (columns)
Nz = 65;     % points in z-direction (rows), Nz ≈ Nx/2 for similar h

% Create grid
x = linspace(x_min, x_max, Nx);
z = linspace(z_min, z_max, Nz);
[X, Z] = meshgrid(x, z);

% Grid spacings (dx ≠ dy in general)
dx = (x_max - x_min) / (Nx - 1);  % x-direction (columns)
dy = (z_max - z_min) / (Nz - 1);  % z-direction (rows)

fprintf('========== Rectangular Domain Test ==========\n');
fprintf('Domain: [%.1f, %.1f] x [%.1f, %.1f]\n', x_min, x_max, z_min, z_max);
fprintf('Grid: Nx = %d, Nz = %d\n', Nx, Nz);
fprintf('Spacing: dx = %.4f, dy = %.4f\n', dx, dy);
fprintf('Aspect ratio dx/dy = %.4f\n', dx/dy);

%% ========== Algorithm Parameters ==========
tol = 0.001;
kmax = 600;
eta = 0.1;
% Landweber parameters
beta = 3;               % regularization strength
Nes_alpha = 5 ;
Nesterov = 0;
niu = 1;                 % Poisson smooth 
duality_r = 2.0;         % duality mapping J_r r value.
tau=1.5;
mu_0 = 1.8*(1 - eta - (1+ eta)/tau)/beta;  % step size parameter; c_0 = 1/2beta
mu_1 = 600;               % step size upper bound
backCond = 1;             % background value

% Create save directory (after beta is defined)
save_dir = fullfile(pwd, 'results', sprintf('%s_%s_beta%.2f', experiment_name, regularization_type, beta));
if save_figures && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Fixed step size (if needed)
use_fixed_alpha = false;
fixed_alpha = 0.01;

%% ========== Exact Solution Configuration ==========
% Define c_exact on rectangular domain
I = Nz;  % number of rows (z-direction)
J = Nx;  % number of columns (x-direction)

% Base layer
c_exact = ones(I, J);
c_min = 0.5;

% Patch 1: central ellipse-like region
% Adjusted to fit rectangular domain
patch1_enabled = true;
if patch1_enabled
    mask1 = (X > -0.8 & X < 0.8 & Z > -0.4 & Z < 0.4);
    c_exact(mask1) = c_exact(mask1) + 1.5;
end

% Patch 2: left region
patch2_enabled = true;
if patch2_enabled
    mask2 = (X > -1.8 & X < -1.2 & Z > -0.6 & Z < 0.6);
    c_exact(mask2) = c_exact(mask2) + 0.8;
end

% Patch 3: right region
patch3_enabled = true;
if patch3_enabled
    mask3 = (X > 1.2 & X < 1.8 & Z > -0.6 & Z < 0.6);
    c_exact(mask3) = c_exact(mask3) + 0.8;
end

c_exact = max(c_exact, c_min);  % ensure positivity

%% ========== Source Points Configuration ==========
% Distribute source points on the rectangular grid
% Adjust source positions to fit the domain

source_grid_x = 12;  % sources in x-direction
source_grid_z = 6;  % sources in z-direction (fewer due to smaller extent)

fixed_pt_list = [];
% Compute source positions in grid indices
source_spacing_i = floor((I - 2) / (source_grid_z + 1));
source_spacing_j = floor((J - 2) / (source_grid_x + 1));

for m = 1:source_grid_z
    for n = 1:source_grid_x
        i_pos = 1 + m * source_spacing_i;
        j_pos = 1 + n * source_spacing_j;
        % TravelTime_solver expects format: [val, j, i]
        % where j = column index (x), i = row index (z)
        fixed_pt_list = [fixed_pt_list; 0, j_pos, i_pos];
    end
end

num_sources = size(fixed_pt_list, 1);
fprintf('Number of sources: %d (%d x %d grid)\n', num_sources, source_grid_z, source_grid_x);


%% ========== observe data (clean) ==========
Tstar_all = zeros(I, J, num_sources);
for p_num = 1:num_sources
    Tstar_all(:,:,p_num) = TravelTime_solver(c_exact, fixed_pt_list(p_num,:), dx, dy, I, J);
end


%% ========== add impulse noise on boundary ==========
p_imp = 0.00;   % 边界点中有 2% 被污染（可调）
A_rel = 0.10;   % 脉冲幅度 = A_rel * (boundary range)（可调）

Tstar_noisy = Tstar_all;
rng_seed = 2025;
rng(rng_seed);
bmask = false(I,J);
bmask(1,:) = true; bmask(I,:) = true; bmask(:,1) = true; bmask(:,J) = true;
bidx = find(bmask);

for p_num = 1:num_sources
    T0 = Tstar_all(:,:,p_num);

    Tb = T0(bidx);
    A  = A_rel * (max(Tb) - min(Tb) + eps);   % 每个 source 自适应幅度

    pick = bidx(rand(numel(bidx),1) < p_imp); % 选中要加脉冲的边界点
    sgn  = sign(randn(numel(pick),1));        % ±1 随机

    T1 = T0;
    T1(pick) = T1(pick) + A .* sgn;           % “尖峰”脉冲：直接加/减大值

    Tstar_noisy(:,:,p_num) = T1;
end

% calculate the noise level 
noise_level = 0;
parfor p_num = 1:size(fixed_pt_list, 1)
    T = TravelTime_solver(c_exact, fixed_pt_list(p_num, :), dx, dy, I, J);
    T_s = Tstar_noisy(:,:,p_num);
    resn_Lr_ = LrNormBoundary(T, T_s, dx, dy, duality_r)^duality_r;
    noise_level = noise_level + resn_Lr_;
end

noise_level = noise_level^(1/duality_r);


%% ========== Initialization ==========
% Initial guess c0 using elliptic smoothing
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);

% Initialize xi based on regularization type
switch regularization_type
    case 'L2'
        xi = c0/beta;

    case 'L1'
        q0 = sign(c0 - backCond);
        xi = c0/beta + q0;

    case 'TV'
        w1 = zeros(size(c0)); 
        w2 = zeros(size(c0));
        [~, w1, w2] = TV_PDHG_host(w1, w2, c0, beta, 100, 1e-6);
        div_w = ([w1(:,1), w1(:,2:end)-w1(:,1:end-1)] + ...
                 [w2(1,:); w2(2:end,:)-w2(1:end-1,:)]);
        q0 = -div_w;
        xi = c0/beta +  q0;
end

% Compute initial c from xi to ensure consistency
c = apply_regularization(xi, regularization_type, beta, backCond, c_min, []);
c0 = c;
xi_n = xi;
zeta = 0;

%% ========== Main Iteration ==========
energy = 1e9;
resn_set = [];
alpha_set = [];
dual_norm_set = [];
c_error_set = [];

fprintf('\n========== Starting Landweber Iteration ==========\n');
fprintf('Regularization: %s, beta = %.2f\n', regularization_type, beta);

for k = 1:kmax
    
    energy_p = 0;
    cstar = 0;
    resn_Lr = 0; 
    if Nesterov
        Nes_lambda  = k/(k + Nes_alpha) ;
    else
        Nes_lambda = 0;
    end

    zeta = xi + Nes_lambda * (xi - xi_n);
    
    parfor p_num = 1:size(fixed_pt_list, 1)
        T = TravelTime_solver(c, fixed_pt_list(p_num, :), dx, dy, I, J);
        T_s = Tstar_noisy(:,:,p_num);
        % T_star = TravelTime_solver(c_exact, fixed_pt_list(p_num, :), dx, dy, I, J);
        resn_Lr_ = LrNormBoundary(T, T_s, dx, dy, duality_r)^duality_r;
        resn_Lr = resn_Lr + resn_Lr_;
        energy_p = energy_p + EnergyFun(T, T_s, dx, dy);
        cstar = cstar + cStarSolver(T, T_s, duality_r, niu, dx, dy, I, J, c);
    end
    
    if energy_p < tol
        fprintf('Converged at iteration %d\n', k);
        break
    end   
    resn_Lr = (resn_Lr)^(1/duality_r);
    if resn_Lr <= tau* noise_level
        fprintf('Iterative ended at iteration %d\n', k);
        break
    end 
        
    energy = [energy, energy_p];
    resn_set = [resn_set, resn_Lr];
    
    % Compute dual norm
    g = cstar;  
    norm_dual_sq = sum(g(:).^2) * dx * dy;  % ||g||_L2^2
    dual_norm_set = [dual_norm_set, norm_dual_sq];
    
    % Compute step size alpha
    if use_fixed_alpha
        alpha = fixed_alpha;
    else
        alpha = min(mu_0 * resn_Lr^(2*(duality_r - 1) ) / max(norm_dual_sq, 1e-12), mu_1) * resn_Lr^(2-duality_r);
    end
    alpha_set = [alpha_set, alpha];
    
    % Update xi
    xi_n = xi;
    xi = zeta + alpha * cstar;
    
    % Apply Regularization
    c = apply_regularization(xi, regularization_type, beta, backCond, c_min, []);
    c_error = norm(c - c_exact, 'fro');
    c_error_set = [c_error_set, c_error];
    % Print progress
    if mod(k, 10) == 0
        fprintf('Iter %4d | Energy = %.3e | Residual = %.3e | Error = %.3e |alpha = %.3e\n', ...
                k, energy(k+1), resn_set(k), c_error_set(k), alpha_set(k));
    end
end

total_iterations = k;
elapsed_time = toc;

%% ========== Results Summary ==========
final_error = norm(c - c_exact, 'fro') * sqrt(dx * dy);
fprintf('\n========== Results Summary ==========\n');
fprintf('Domain: [%.1f, %.1f] x [%.1f, %.1f]\n', x_min, x_max, z_min, z_max);
fprintf('Grid: %d x %d (dx=%.4f, dy=%.4f)\n', I, J, dx, dy);
fprintf('Regularization type: %s\n', regularization_type);
fprintf('Total iterations: %d\n', total_iterations);
fprintf('Final L2 error: %.6e\n', final_error);
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

%% ========== Figure 1: Solution Comparison ==========
fig1 = figure('Position', [100, 100, 1400, 600]);
sgtitle(sprintf('Rectangular Domain: Landweber with %s (\\beta=%.2f, dx=%.3f, dy=%.3f)', ...
        regularization_type, beta, dx, dy), 'FontSize', 14, 'FontWeight', 'bold');

% Consistent colorbar limits
cmin_sol = min([c_exact(:); c(:); c0(:)]);
cmax_sol = max([c_exact(:); c(:); c0(:)]);

subplot(2, 3, 1)
imagesc(x, z, c_exact)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title('c_{exact}', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_sol, cmax_sol])
colorbar
axis xy equal tight

subplot(2, 3, 2)
imagesc(x, z, c0)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title('c_0 (Initial Guess)', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_sol, cmax_sol])
colorbar
axis xy equal tight

subplot(2, 3, 3)
imagesc(x, z, c)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title(sprintf('c_{solution} (%s, \\beta=%.2f)', regularization_type, beta), 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_sol, cmax_sol])
colorbar
axis xy equal tight

subplot(2, 3, 4)
imagesc(x, z, abs(c - c_exact))
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title('Absolute Error', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis xy equal tight

subplot(2, 3, 5)
% Cross-section at z = 0
mid_i = ceil(I/2);
plot(x, c_exact(mid_i, :), '-', 'LineWidth', 2, 'DisplayName', 'c_{exact}')
hold on
plot(x, c(mid_i, :), '--', 'LineWidth', 2, 'DisplayName', 'c_{solution}')
plot(x, c0(mid_i, :), '-.', 'LineWidth', 1.5, 'DisplayName', 'c_0')
hold off
legend('Location', 'best')
xlabel('x', 'FontSize', 12)
ylabel('c', 'FontSize', 12)
title('Cross-section at z = 0', 'FontWeight', 'bold', 'FontSize', 12)
grid on

subplot(2, 3, 6)
% Cross-section at x = 0
mid_j = ceil(J/2);
plot(z, c_exact(:, mid_j), '-', 'LineWidth', 2, 'DisplayName', 'c_{exact}')
hold on
plot(z, c(:, mid_j), '--', 'LineWidth', 2, 'DisplayName', 'c_{solution}')
plot(z, c0(:, mid_j), '-.', 'LineWidth', 1.5, 'DisplayName', 'c_0')
hold off
legend('Location', 'best')
xlabel('z', 'FontSize', 12)
ylabel('c', 'FontSize', 12)
title('Cross-section at x = 0', 'FontWeight', 'bold', 'FontSize', 12)
grid on

if save_figures
    saveas(fig1, fullfile(save_dir, 'solution_comparison.png'));
    saveas(fig1, fullfile(save_dir, 'solution_comparison.fig'));
    fprintf('Saved: solution_comparison.png\n');
end

%% ========== Figure 2: Convergence Analysis ==========
resn_set_ = sqrt(resn_set);

fig2 = figure('Position', [100, 100, 1500, 450]);  % 宽一点
sgtitle(sprintf('Convergence - Rectangular Domain (%s, \\beta=%.2f)', ...
    regularization_type, beta), 'FontSize', 14, 'FontWeight', 'bold');

% =======================
% 1) Energy
% =======================
subplot(1, 4, 1)
y1 = energy(2:end);
x1 = 1:numel(y1);

semilogy(x1, y1, 'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410])
grid on
title('Energy (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('Energy', 'FontSize', 11)

% ---- mark minimum ----
[emin, idx1] = min(y1);
hold on
plot(x1(idx1), emin, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 6)
text(x1(idx1), emin, sprintf('  min=%.3e @%d', emin, x1(idx1)), ...
    'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom')
hold off


% =======================
% 2) Residual
% =======================
subplot(1, 4, 2)
y2 = resn_set_;
x2 = 1:numel(y2);

semilogy(x2, y2, 'LineWidth', 1.5, 'Color', [0.8500, 0.3250, 0.0980])
grid on
title('Residual ||r|| (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('||r||', 'FontSize', 11)

% ---- mark minimum ----
[rmin, idx2] = min(y2);
hold on
plot(x2(idx2), rmin, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 6)
text(x2(idx2), rmin, sprintf('  min=%.3e @%d', rmin, x2(idx2)), ...
    'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom')
hold off


% =======================
% 3) Step size alpha (no min marker)
% =======================
subplot(1, 4, 3)
plot(alpha_set, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880])
grid on
title('Step Size \alpha', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('\alpha', 'FontSize', 11)


% =======================
% 4) c-error
% =======================
subplot(1, 4, 4)
y4 = c_error_set;
x4 = 1:numel(y4);

semilogy(x4, y4, 'LineWidth', 1.5, 'Color', [0.4940, 0.1840, 0.5560])
grid on
title('c-error (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('||c-c_{ref}||', 'FontSize', 11)

% ---- mark minimum ----
[cminv, idx4] = min(y4);
hold on
plot(x4(idx4), cminv, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 6)
text(x4(idx4), cminv, sprintf('  min=%.3e @%d', cminv, x4(idx4)), ...
    'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom')
hold off


% =======================
% Save
% =======================
if save_figures
    if ~exist(save_dir,'dir'); mkdir(save_dir); end  % 防止没创建文件夹
    saveas(fig2, fullfile(save_dir, 'convergence.png'));
    saveas(fig2, fullfile(save_dir, 'convergence.fig'));
    fprintf('Saved: convergence.png\n');
end
%% ========== Figure 3: Traveltime Field Example ==========
fig3 = figure('Position', [100, 100, 1000, 400]);
sgtitle(sprintf('Traveltime Field Comparison (%s, \\beta=%.2f)', regularization_type, beta), 'FontSize', 14, 'FontWeight', 'bold');

% Use a source near the center
center_source = [0, ceil(I/2), ceil(J/2)];
T_exact = TravelTime_solver(c_exact, center_source, dx, dy, I, J);
T_approx = TravelTime_solver(c, center_source, dx, dy, I, J);

subplot(1, 3, 1)
contourf(x, z, T_exact, 20)
xlabel('x', 'FontSize', 11)
ylabel('z', 'FontSize', 11)
title('T (exact c)', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis equal tight

subplot(1, 3, 2)
contourf(x, z, T_approx, 20)
xlabel('x', 'FontSize', 11)
ylabel('z', 'FontSize', 11)
title('T (reconstructed c)', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis equal tight

subplot(1, 3, 3)
contourf(x, z, abs(T_exact - T_approx), 20)
xlabel('x', 'FontSize', 11)
ylabel('z', 'FontSize', 11)
title('|T_{exact} - T_{approx}|', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis equal tight

if save_figures
    saveas(fig3, fullfile(save_dir, 'traveltime_comparison.png'));
    saveas(fig3, fullfile(save_dir, 'traveltime_comparison.fig'));
    fprintf('Saved: traveltime_comparison.png\n');
end


%% ========== Figure 4: Observation Data (Clean vs Noisy) + Boundary 1D ==========
fig4 = figure('Position', [100, 100, 1600, 850]);
sgtitle(sprintf('Observation Data Comparison (Clean vs Impulse-Noisy) | Source #%d', 1), ...
        'FontSize', 14, 'FontWeight', 'bold');

% ---- choose source index to show ----
p = 1;  % 1..num_sources

% ---- fetch data ----
Tclean = Tstar_all(:,:,p);
Tnoisy = Tstar_noisy(:,:,p);
D      = Tnoisy - Tclean;

% ---- consistent color limits for clean/noisy ----
climT = [min(Tclean(:)), max(Tclean(:))];

% ===================== (1) Clean full-field =====================
subplot(2,3,1)
imagesc(x, z, Tclean);
axis xy equal tight;
title('Clean Observation: T^* (Full Field)', 'FontWeight', 'bold');
xlabel('x'); ylabel('z');
caxis(climT);
colorbar;

% ===================== (2) Noisy full-field =====================
subplot(2,3,2)
imagesc(x, z, Tnoisy);
axis xy equal tight;
title('Noisy Observation: $\tilde{T}^*$ (Impulse Noise on $\partial\Omega$)', ...
      'FontWeight','bold','Interpreter','latex');
xlabel('x'); ylabel('z');
caxis(climT);
colorbar;

% ===================== (3) Difference full-field =====================
subplot(2,3,3)
imagesc(x, z, D);
axis xy equal tight;
title('$\Delta = \tilde{T}^* - T^*$ (Full Field)', ...
      'FontWeight','bold','Interpreter','latex');
xlabel('x'); ylabel('z');
colorbar;

% ===================== Boundary 1D construction (arclength s) =====================
% Boundary traversal (clockwise):
%   Top:    (i=1,  j=1..J)
%   Right:  (i=2..I, j=J)
%   Bottom: (i=I,  j=J-1..1)
%   Left:   (i=I-1..2, j=1)
% Corners are included only once.

top_c = Tclean(1, 1:J);          top_n = Tnoisy(1, 1:J);
rig_c = Tclean(2:I, J);          rig_n = Tnoisy(2:I, J);
bot_c = Tclean(I, J-1:-1:1);     bot_n = Tnoisy(I, J-1:-1:1);
lef_c = Tclean(I-1:-1:2, 1);     lef_n = Tnoisy(I-1:-1:2, 1);

bc = [top_c(:); rig_c(:); bot_c(:); lef_c(:)];
bn = [top_n(:); rig_n(:); bot_n(:); lef_n(:)];
bd = bn - bc;

% arclength coordinate s with correct physical step sizes (dx on horizontal edges, dy on vertical edges)
nTop = numel(top_c);            % = J
nRig = numel(rig_c);            % = I-1
nBot = numel(bot_c);            % = J-1
nLef = numel(lef_c);            % = I-2

sTop = (0:(nTop-1))' * dx;
sRig = sTop(end) + (1:nRig)' * dy;
sBot = sRig(end) + (1:nBot)' * dx;
sLef = sBot(end) + (1:nLef)' * dy;

s = [sTop; sRig; sBot; sLef];

% segment boundaries (arclength values)
s1 = sTop(end);   % end of Top
s2 = sRig(end);   % end of Right
s3 = sBot(end);   % end of Bottom
% s4 = sLef(end); % end of Left (total perimeter)

% ===================== (4) Boundary 1D curves (clean vs noisy) =====================
subplot(2,3,4)
h1 = plot(s, bc, 'LineWidth', 1.8, 'DisplayName','T^* (clean)'); hold on
h2 =plot(s, bn, '--', 'LineWidth', 1.4, 'DisplayName','\~T^* (impulse-noisy)');

xline(s1, ':', 'Top|Right',    'LabelVerticalAlignment','bottom', 'HandleVisibility','off');
xline(s2, ':', 'Right|Bottom', 'LabelVerticalAlignment','bottom', 'HandleVisibility','off');
xline(s3, ':', 'Bottom|Left',  'LabelVerticalAlignment','bottom', 'HandleVisibility','off');

% legend('show','Location','best');
legend([h1 h2], {'$T^*$ (clean)', '$\tilde{T}^*$ (impulse-noisy)'}, ...
       'Location','best','Interpreter','latex');
hold off

% ===================== (5) Boundary difference (stem) =====================
subplot(2,3,5)
stem(s, bd, 'filled');
grid on;
xlabel('Boundary arclength s (same ordering)');
ylabel('$\Delta(s) = \tilde{T}^*(s) - T^*(s)$', 'Interpreter','latex');
title('Boundary Noise (Impulse Outliers) | \Delta on \partial\Omega', 'FontWeight','bold');

xline(s1, ':'); xline(s2, ':'); xline(s3, ':');

% ===================== (6) Histogram of boundary noise =====================
subplot(2,3,6)
histogram(bd, 60);
grid on;
xlabel('\Delta values on \partial\Omega');
ylabel('Count');
title('Histogram of Boundary Noise (\Delta)', 'FontWeight','bold');

% ===================== Save =====================
if save_figures
    saveas(fig4, fullfile(save_dir, 'contour_comparison.png'));
    saveas(fig4, fullfile(save_dir, 'contour_comparison.fig'));
    fprintf('Saved: contour_comparison.png\n');
end

%% ========== Save Experiment Data ==========
if save_figures
    experiment_data.regularization_type = regularization_type;
    experiment_data.experiment_name = experiment_name;
    experiment_data.domain = [x_min, x_max, z_min, z_max];
    experiment_data.Nx = Nx;
    experiment_data.Nz = Nz;
    experiment_data.I = I;
    experiment_data.J = J;
    experiment_data.dx = dx;
    experiment_data.dy = dy;
    experiment_data.beta = beta;
    experiment_data.tol = tol;
    experiment_data.kmax = kmax;
    experiment_data.total_iterations = total_iterations;
    experiment_data.final_error = final_error;
    experiment_data.elapsed_time = elapsed_time;
    experiment_data.num_sources = num_sources;
    experiment_data.energy = energy;
    experiment_data.resn_set = resn_set;
    experiment_data.alpha_set = alpha_set;
    experiment_data.c_exact = c_exact;
    experiment_data.c_solution = c;
    experiment_data.c0 = c0;
    experiment_data.x = x;
    experiment_data.z = z;


    % noise metadata
    experiment_data.noise.type     = 'impulse_boundary_additive'; % 你用的是“加/减尖峰”
    experiment_data.noise.p_imp    = p_imp;
    experiment_data.noise.A_rel    = A_rel;
    experiment_data.noise.rng_seed = rng_seed;
    experiment_data.noise.boundary_only = true;   % 语义：只对边界加噪声（如果你确实是这样做的）

    experiment_data.duality_r = duality_r;              % r>1
    
    save(fullfile(save_dir, 'experiment_data.mat'), 'experiment_data');
    fprintf('Saved: experiment_data.mat\n');
    fprintf('\nAll results saved to: %s\n', save_dir);
end

fprintf('\n========== Rectangular Domain Test Complete ==========\n');
