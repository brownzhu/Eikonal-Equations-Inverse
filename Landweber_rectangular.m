% Landweber iteration for RECTANGULAR (non-square) domain
% Tests the solver with dx ≠ dy
% Supports L1, L2, TV regularization
clc; clear; close all
tic
format long

%% ========== Experiment Configuration ==========
% Regularization type: 'L1', 'L2', 'TV'
% regularization_type = 'L2';
regularization_type = 'L2';
% regularization_type = 'L1';

% Whether to save figures
save_figures = true;

% Experiment name
experiment_name = 'rectangular_domain';

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
kmax = 1500;

% Landweber parameters
beta = 5.0;               % regularization strength
mu_0 = 0.8*(1 - 1/1.05);  % step size parameter
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

source_grid_x = 6;  % sources in x-direction
source_grid_z = 3;  % sources in z-direction (fewer due to smaller extent)

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

%% ========== Initialization ==========
niu = 1;

% Initial guess c0 using elliptic smoothing
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);

% Initialize xi based on regularization type
switch regularization_type
    case 'L2'
        xi = c0;

    case 'L1'
        q0 = sign(c0 - backCond);
        xi = c0 + beta * q0;

    case 'TV'
        w1 = zeros(size(c0)); 
        w2 = zeros(size(c0));
        [~, w1, w2] = TV_PDHG_host(w1, w2, c0, beta, 100, 1e-6);
        div_w = ([w1(:,1), w1(:,2:end)-w1(:,1:end-1)] + ...
                 [w2(1,:); w2(2:end,:)-w2(1:end-1,:)]);
        q0 = -div_w;
        xi = c0 + beta * q0;
end

% Compute initial c from xi to ensure consistency
c = apply_regularization(xi, regularization_type, beta, backCond, c_min, []);
c0 = c;

%% ========== Main Iteration ==========
energy = 1e9;
resn_set = [];
alpha_set = [];
dual_norm_set = [];

fprintf('\n========== Starting Landweber Iteration ==========\n');
fprintf('Regularization: %s, beta = %.2f\n', regularization_type, beta);

for k = 1:kmax
    
    energy_p = 0;
    cstar = 0;
    resn_power = 0; 
    
    parfor p_num = 1:size(fixed_pt_list, 1)
        T = TravelTime_solver(c, fixed_pt_list(p_num, :), dx, dy, I, J);
        T_star = TravelTime_solver(c_exact, fixed_pt_list(p_num, :), dx, dy, I, J);
        resn_power_ = L2NormBoundary(T, T_star, dx, dy)^2;
        resn_power = resn_power + resn_power_;
        energy_p = energy_p + EnergyFun(T, T_star, dx, dy);
        cstar = cstar + cStarSolver(T, T_star, dx, dy, I, J, c);
    end
    
    if energy_p < tol
        fprintf('Converged at iteration %d\n', k);
        break
    end

    energy = [energy, energy_p];
    resn_set = [resn_set, resn_power];
    
    % Compute dual norm
    g = cstar;  
    norm_dual_sq = sum(g(:).^2) * dx * dy;  % ||g||_L2^2
    dual_norm_set = [dual_norm_set, norm_dual_sq];
    
    % Compute step size alpha
    if use_fixed_alpha
        alpha = fixed_alpha;
    else
        alpha = min(mu_0 * resn_power / max(norm_dual_sq, 1e-12), mu_1);
    end
    alpha_set = [alpha_set, alpha];
    
    % Update xi
    xi = xi + alpha * cstar;
    
    % Apply Regularization
    c = apply_regularization(xi, regularization_type, beta, backCond, c_min, []);
    
    % Print progress
    if mod(k, 10) == 0
        fprintf('Iter %4d | Energy = %.6e | Residual Sq. = %.6e | alpha = %.3e\n', ...
                k, energy(k+1), resn_set(k), alpha_set(k));
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
fig2 = figure('Position', [100, 100, 1200, 500]);
sgtitle(sprintf('Convergence - Rectangular Domain (%s, \\beta=%.2f)', regularization_type, beta), ...
        'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 3, 1)
semilogy(energy(2:end), 'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410])
grid on
title('Energy (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('Energy', 'FontSize', 11)

subplot(1, 3, 2)
semilogy(resn_set, 'LineWidth', 1.5, 'Color', [0.8500, 0.3250, 0.0980])
grid on
title('Residual ||r||^2 (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('||r||^2', 'FontSize', 11)

subplot(1, 3, 3)
plot(alpha_set, 'LineWidth', 1.5, 'Color', [0.4660, 0.6740, 0.1880])
grid on
title('Step Size \alpha', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('\alpha', 'FontSize', 11)

if save_figures
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
    
    save(fullfile(save_dir, 'experiment_data.mat'), 'experiment_data');
    fprintf('Saved: experiment_data.mat\n');
    fprintf('\nAll results saved to: %s\n', save_dir);
end

fprintf('\n========== Rectangular Domain Test Complete ==========\n');
