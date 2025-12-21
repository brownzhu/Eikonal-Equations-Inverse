% 1st order Landweber iteration with regularization
% Supports L1, L2, TV regularization for piecewise solution reconstruction
clc; clear; close all
tic
format long

%% ========== Experiment Configuration ==========
% Regularization type: 'L1', 'L2', 'TV'
regularization_type = 'TV';
% regularization_type = 'L2';
% regularization_type = 'L1';
% Whether to save figures
save_figures = true;
% save_figures = false;

% Experiment name (used for saving files)
experiment_name = 'piecewise_1patch';

% Create save directory
save_dir = fullfile(pwd, 'results', [experiment_name '_' regularization_type]);
if save_figures && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% ========== Algorithm Parameters ==========
tol = 0.001;
kmax = 2000;
N = 129;
x = linspace(-1, 1, N);
z = linspace(-1, 1, N);
[X, Z] = meshgrid(x, z);

% Landweber parameters
beta = 10.0;              % regularization strength
mu_0 = 0.8*(1 - 1/1.05);  % step size parameter
mu_1 = 600;               % step size upper bound
backCond = 1;             % background value

% Fixed step size (if needed)
use_fixed_alpha = false;
% use_fixed_alpha = true;
fixed_alpha = 0.01;

%% ========== Piecewise Solution Configuration ==========
% Base layer
c_exact = ones(N);
c_min = 0.5;

% patch 1: central rectangle (positive perturbation)
patch1_enabled = true;
if patch1_enabled
    mask1 = (X > -0.4 & X < 0.4 & Z > -0.2 & Z < 0.2);
    c_exact(mask1) = c_exact(mask1) + 1.5;
end

% patch 2: upper-left rectangle (negative perturbation)
patch2_enabled = false;
if patch2_enabled
    mask2 = (X > -0.9 & X < -0.5 & Z > 0.4 & Z < 0.8);
    c_exact(mask2) = c_exact(mask2) - 0.3;
end

% patch 3: lower-right rectangle (positive perturbation)
patch3_enabled = false;
if patch3_enabled
    mask3 = (X > 0.3 & X < 0.8 & Z > -0.8 & Z < -0.4);
    c_exact(mask3) = c_exact(mask3) + 1.0;
end

c_exact = max(c_exact, c_min);  % ensure positivity (important)

%% ========== Source Points Configuration ==========
% Source point configuration
source_grid = 4;  % 4x4 = 16 source points
fixed_pt_list = [];
for m = 1:source_grid
    for n = 1:source_grid
        fixed_pt_list = [fixed_pt_list; 0, m*25, n*25];
    end
end
num_sources = size(fixed_pt_list, 1);
fprintf('Number of sources: %d\n', num_sources);

%% ========== Initialization ==========
I = N;
J = N;
dx = (x(end)-x(1)) / (I-1); 
dy = (z(end) - z(1)) / (J-1);

niu = 1;
% (I - niu laplace) c0 = 0, with exact Dirichlet boundary
% initial guess c0
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);
% c = c0;

% Initialize xi using regularization inverse
% xi_0 = nabla \Theta^* (c_0), i.e., find xi such that apply_regularization(xi) = c0
% For simplicity, we use c0 as initial guess for xi
switch regularization_type
    case 'L2'
        xi = c0;

    case 'L1'
        q0  = sign(c0 - backCond);
        xi = c0 + beta * q0;

    case 'TV'
        w1 = zeros(size(c0)); 
        w2 = zeros(size(c0));
        [~, w1, w2] = TV_PDHG_host(w1, w2, c0, beta, 100, 1e-6);

        div_w = ([w1(:,1), w1(:,2:end)-w1(:,1:end-1)] + ...
                 [w2(1,:); w2(2:end,:)-w2(1:end-1,:)]);
        q0  = -div_w;
        xi = c0 + beta * q0;
end
% Compute initial c from xi to ensure consistency
c = apply_regularization(xi, regularization_type, beta, backCond, c_min, N);
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
        cstar = cstar + cStarSolver(T, T_star, dx, I, J, c);
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
    
    % Apply Regularization using the unified function
    c = apply_regularization(xi, regularization_type, beta, backCond, c_min, N);
    
    % Print progress
    if mod(k, 10) == 0
        fprintf('Iter %4d | Energy = %.6e | Residual Sq. = %.6e | alpha = %.3e\n', ...
                k, energy(k+1), resn_set(k), alpha_set(k));
    end
end

total_iterations = k;
elapsed_time = toc;

%% ========== Results Summary ==========
final_error = norm(c-c_exact)*dx*dx;
fprintf('\n========== Results Summary ==========\n');
fprintf('Regularization type: %s\n', regularization_type);
fprintf('Total iterations: %d\n', total_iterations);
fprintf('Final L2 error: %.6e\n', final_error);
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

%% ========== Figure 1: Solution Comparison ==========
fig1 = figure('Position', [100, 100, 1200, 900]);
sgtitle(sprintf('Landweber with %s Regularization (\\beta=%.1f, \\alpha=%.3f)', ...
        regularization_type, beta, fixed_alpha), 'FontSize', 14, 'FontWeight', 'bold');

% Compute consistent colorbar limits for c_exact and c_solution
cmin_sol = min([c_exact(:); c(:); c0(:)]);
cmax_sol = max([c_exact(:); c(:); c0(:)]);

subplot(2, 2, 1)
mesh(x, z, c_exact)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title('c_{exact}', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_sol, cmax_sol])
colorbar

subplot(2, 2, 2)
mesh(x, z, c0)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title('c_0 (Initial Guess)', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_sol, cmax_sol])
colorbar

subplot(2, 2, 3)
mesh(x, z, c)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title(sprintf('c_{solution} (%s)', regularization_type), 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_sol, cmax_sol])
colorbar

subplot(2, 2, 4)
mesh(x, z, (c-c_exact)./c_exact)
xlabel('x', 'FontSize', 12)
ylabel('z', 'FontSize', 12)
title('Relative Error', 'FontWeight', 'bold', 'FontSize', 12)
colorbar

if save_figures
    saveas(fig1, fullfile(save_dir, 'solution_comparison.png'));
    saveas(fig1, fullfile(save_dir, 'solution_comparison.fig'));
    fprintf('Saved: solution_comparison.png\n');
end

%% ========== Figure 2: Convergence Analysis ==========
fig2 = figure('Position', [100, 100, 1200, 900]);
sgtitle(sprintf('Convergence Analysis - %s Regularization', regularization_type), ...
        'FontSize', 14, 'FontWeight', 'bold');

subplot(2, 2, 1)
semilogy(energy(2:end), 'LineWidth', 1.5, 'Color', [0, 0.4470, 0.7410])
grid on
title('Energy (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('Energy', 'FontSize', 11)

subplot(2, 2, 2)
semilogy(resn_set(1:end), 'LineWidth', 1.5, 'Color', [0.8500, 0.3250, 0.0980])
grid on
title('Residual Power (log scale)', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('Iteration', 'FontSize', 11)
ylabel('||r||^2', 'FontSize', 11)

subplot(2, 2, 3)
mid_idx = ceil(N/2);
plot(z, c0(mid_idx, :), '-.', 'LineWidth', 1.5, 'DisplayName', 'c_0')
hold on
plot(z, c(mid_idx, :), '--', 'LineWidth', 2, 'DisplayName', 'c_{numerical}')
plot(z, c_exact(mid_idx, :), '-', 'LineWidth', 1.5, 'DisplayName', 'c_{exact}')
hold off
legend('Location', 'best', 'FontSize', 10)
title('Cross-section at x = 0', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('z', 'FontSize', 11)
ylabel('c', 'FontSize', 11)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
grid on

subplot(2, 2, 4)
plot(x, c0(:, mid_idx), '-.', 'LineWidth', 1.5, 'DisplayName', 'c_0')
hold on
plot(x, c(:, mid_idx), '--', 'LineWidth', 2, 'DisplayName', 'c_{numerical}')
plot(x, c_exact(:, mid_idx), '-', 'LineWidth', 1.5, 'DisplayName', 'c_{exact}')
hold off
legend('Location', 'best', 'FontSize', 10)
title('Cross-section at z = 0', 'FontWeight', 'bold', 'FontSize', 12)
xlabel('x', 'FontSize', 11)
ylabel('c', 'FontSize', 11)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
grid on

if save_figures
    saveas(fig2, fullfile(save_dir, 'convergence_analysis.png'));
    saveas(fig2, fullfile(save_dir, 'convergence_analysis.fig'));
    fprintf('Saved: convergence_analysis.png\n');
end

%% ========== Figure 3: Contour Comparison ==========
fig3 = figure('Position', [100, 100, 1200, 400]);
sgtitle(sprintf('Contour Comparison - %s Regularization', regularization_type), ...
        'FontSize', 14, 'FontWeight', 'bold');

% Consistent colorbar limits for contour plots
cmin_contour = min([c_exact(:); c(:)]);
cmax_contour = max([c_exact(:); c(:)]);

subplot(1, 3, 1)
contourf(x, z, c_exact, 20)
xlabel('x', 'FontSize', 11)
ylabel('z', 'FontSize', 11)
title('c_{exact}', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_contour, cmax_contour])
colorbar
axis equal tight

subplot(1, 3, 2)
contourf(x, z, c, 20)
xlabel('x', 'FontSize', 11)
ylabel('z', 'FontSize', 11)
title(sprintf('c_{solution} (%s)', regularization_type), 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_contour, cmax_contour])
colorbar
axis equal tight

subplot(1, 3, 3)
contourf(x, z, abs(c - c_exact), 20)
xlabel('x', 'FontSize', 11)
ylabel('z', 'FontSize', 11)
title('Absolute Error', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis equal tight

if save_figures
    saveas(fig3, fullfile(save_dir, 'contour_comparison.png'));
    saveas(fig3, fullfile(save_dir, 'contour_comparison.fig'));
    fprintf('Saved: contour_comparison.png\n');
end

%% ========== Save Experiment Data ==========
if save_figures
    % Save all experiment parameters and results
    experiment_data.regularization_type = regularization_type;
    experiment_data.experiment_name = experiment_name;
    experiment_data.N = N;
    experiment_data.beta = beta;
    experiment_data.alpha = fixed_alpha;
    experiment_data.use_fixed_alpha = use_fixed_alpha;
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

fprintf('\n========== Experiment Complete ==========\n');