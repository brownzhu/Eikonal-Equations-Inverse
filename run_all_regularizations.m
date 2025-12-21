% Run Landweber iteration with all three regularization methods
% for piecewise solution experiments
% This script runs L1, L2, and TV regularization separately and compares results

clc; clear; close all
format long

%% ========== Common Parameters ==========
tol = 0.001;
kmax = 1500;
N = 129;
x = linspace(-1, 1, N);
z = linspace(-1, 1, N);
[X, Z] = meshgrid(x, z);

% Landweber parameters
beta = 10.0;
mu_0 = 0.8*(1 - 1/1.05);
mu_1 = 600;
backCond = 1;
c_min = 0.5;

% Fixed step size
use_fixed_alpha = true;
fixed_alpha = 0.01;

% Experiment configuration
experiment_name = 'piecewise_comparison';
save_figures = true;

% Create main save directory
main_save_dir = fullfile(pwd, 'results', experiment_name);
if save_figures && ~exist(main_save_dir, 'dir')
    mkdir(main_save_dir);
end

%% ========== Piecewise Solution Configuration ==========
c_exact = ones(N);

% patch 1: central rectangle (positive perturbation)
mask1 = (X > -0.4 & X < 0.4 & Z > -0.2 & Z < 0.2);
c_exact(mask1) = c_exact(mask1) + 1.5;

% Uncomment for more patches
% mask2 = (X > -0.9 & X < -0.5 & Z > 0.4 & Z < 0.8);
% c_exact(mask2) = c_exact(mask2) - 0.3;
% 
% mask3 = (X > 0.3 & X < 0.8 & Z > -0.8 & Z < -0.4);
% c_exact(mask3) = c_exact(mask3) + 1.0;

c_exact = max(c_exact, c_min);

%% ========== Source Points ==========
source_grid = 4;
fixed_pt_list = [];
for m = 1:source_grid
    for n = 1:source_grid
        fixed_pt_list = [fixed_pt_list; 0, m*25, n*25];
    end
end

%% ========== Grid Setup ==========
I = N; J = N;
dx = (x(end)-x(1)) / (I-1);
dy = (z(end)-z(1)) / (J-1);

niu = 1;
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);

%% ========== Run Three Regularization Methods ==========
regularization_types = {'L1', 'L2', 'TV'};
results = struct();

for reg_idx = 1:length(regularization_types)
    reg_type = regularization_types{reg_idx};
    fprintf('\n\n========================================\n');
    fprintf('Running %s Regularization\n', reg_type);
    fprintf('========================================\n');
    
    tic
    
    % Initialize
    c = c0;
    xi = c0;
    % Compute initial c from xi to ensure consistency
    c = apply_regularization(xi, reg_type, beta, backCond, c_min, N);
    energy = 1e9;
    resn_set = [];
    alpha_set = [];
    
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
        
        % Step size
        if use_fixed_alpha
            alpha = fixed_alpha;
        else
            g = cstar;
            norm_dual_sq = sum(g(:).^2) * dx * dy;
            alpha = min(mu_0 * resn_power / max(norm_dual_sq, 1e-12), mu_1);
        end
        alpha_set = [alpha_set, alpha];
        
        % Update xi
        xi = xi + alpha * cstar;
        
        % Apply regularization
        c = apply_regularization(xi, reg_type, beta, backCond, c_min, N);
        
        if mod(k, 50) == 0
            fprintf('%s | Iter %4d | Energy = %.6e\n', reg_type, k, energy(k+1));
        end
    end
    
    elapsed = toc;
    final_error = norm(c - c_exact) * dx * dx;
    
    % Store results
    results.(reg_type).c = c;
    results.(reg_type).energy = energy;
    results.(reg_type).resn_set = resn_set;
    results.(reg_type).iterations = k;
    results.(reg_type).final_error = final_error;
    results.(reg_type).time = elapsed;
    
    fprintf('%s completed: %d iterations, error = %.6e, time = %.2fs\n', ...
            reg_type, k, final_error, elapsed);
end

%% ========== Comparison Plots ==========

% Compute consistent colorbar limits for all solutions
cmin_all = min([c_exact(:); c0(:); results.L1.c(:); results.L2.c(:); results.TV.c(:)]);
cmax_all = max([c_exact(:); c0(:); results.L1.c(:); results.L2.c(:); results.TV.c(:)]);

% Figure 1: Solution comparison (mesh plots)
fig1 = figure('Position', [50, 50, 1400, 800]);
sgtitle('Solution Comparison: L1 vs L2 vs TV Regularization', 'FontSize', 14, 'FontWeight', 'bold');

subplot(2, 3, 1)
mesh(x, z, c_exact)
title('c_{exact}', 'FontWeight', 'bold')
xlabel('x'); ylabel('z');
caxis([cmin_all, cmax_all]); colorbar

subplot(2, 3, 2)
mesh(x, z, c0)
title('c_0 (Initial)', 'FontWeight', 'bold')
xlabel('x'); ylabel('z');
caxis([cmin_all, cmax_all]); colorbar

subplot(2, 3, 3)
% Empty or add info text
axis off
text(0.5, 0.7, sprintf('Experiment: %s', experiment_name), ...
     'HorizontalAlignment', 'center', 'FontSize', 12);
text(0.5, 0.5, sprintf('Grid: %dx%d, Sources: %d', N, N, size(fixed_pt_list,1)), ...
     'HorizontalAlignment', 'center', 'FontSize', 11);
text(0.5, 0.3, sprintf('\\beta = %.1f, \\alpha = %.3f', beta, fixed_alpha), ...
     'HorizontalAlignment', 'center', 'FontSize', 11);

subplot(2, 3, 4)
mesh(x, z, results.L1.c)
title(sprintf('L1: err=%.4e', results.L1.final_error), 'FontWeight', 'bold')
xlabel('x'); ylabel('z');
caxis([cmin_all, cmax_all]); colorbar

subplot(2, 3, 5)
mesh(x, z, results.L2.c)
title(sprintf('L2: err=%.4e', results.L2.final_error), 'FontWeight', 'bold')
xlabel('x'); ylabel('z');
caxis([cmin_all, cmax_all]); colorbar

subplot(2, 3, 6)
mesh(x, z, results.TV.c)
title(sprintf('TV: err=%.4e', results.TV.final_error), 'FontWeight', 'bold')
xlabel('x'); ylabel('z');
caxis([cmin_all, cmax_all]); colorbar

if save_figures
    saveas(fig1, fullfile(main_save_dir, 'comparison_mesh.png'));
    saveas(fig1, fullfile(main_save_dir, 'comparison_mesh.fig'));
end

% Figure 2: Contour comparison
fig2 = figure('Position', [50, 50, 1400, 600]);
sgtitle('Contour Comparison', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 4, 1)
contourf(x, z, c_exact, 20);
caxis([cmin_all, cmax_all]); colorbar
title('c_{exact}', 'FontWeight', 'bold')
axis equal tight

subplot(1, 4, 2)
contourf(x, z, results.L1.c, 20);
caxis([cmin_all, cmax_all]); colorbar
title('L1', 'FontWeight', 'bold')
axis equal tight

subplot(1, 4, 3)
contourf(x, z, results.L2.c, 20);
caxis([cmin_all, cmax_all]); colorbar
title('L2', 'FontWeight', 'bold')
axis equal tight

subplot(1, 4, 4)
contourf(x, z, results.TV.c, 20);
caxis([cmin_all, cmax_all]); colorbar
title('TV', 'FontWeight', 'bold')
axis equal tight

if save_figures
    saveas(fig2, fullfile(main_save_dir, 'comparison_contour.png'));
    saveas(fig2, fullfile(main_save_dir, 'comparison_contour.fig'));
end

% Figure 3: Cross-sections
fig3 = figure('Position', [50, 50, 1200, 500]);
sgtitle('Cross-Section Comparison', 'FontSize', 14, 'FontWeight', 'bold');

mid_idx = ceil(N/2);

subplot(1, 2, 1)
plot(z, c_exact(mid_idx, :), 'k-', 'LineWidth', 2, 'DisplayName', 'Exact')
hold on
plot(z, results.L1.c(mid_idx, :), 'r--', 'LineWidth', 1.5, 'DisplayName', 'L1')
plot(z, results.L2.c(mid_idx, :), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'L2')
plot(z, results.TV.c(mid_idx, :), 'b:', 'LineWidth', 2, 'DisplayName', 'TV')
hold off
legend('Location', 'best')
title('Cross-section at x = 0', 'FontWeight', 'bold')
xlabel('z'); ylabel('c')
grid on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

subplot(1, 2, 2)
plot(x, c_exact(:, mid_idx), 'k-', 'LineWidth', 2, 'DisplayName', 'Exact')
hold on
plot(x, results.L1.c(:, mid_idx), 'r--', 'LineWidth', 1.5, 'DisplayName', 'L1')
plot(x, results.L2.c(:, mid_idx), 'g-.', 'LineWidth', 1.5, 'DisplayName', 'L2')
plot(x, results.TV.c(:, mid_idx), 'b:', 'LineWidth', 2, 'DisplayName', 'TV')
hold off
legend('Location', 'best')
title('Cross-section at z = 0', 'FontWeight', 'bold')
xlabel('x'); ylabel('c')
grid on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

if save_figures
    saveas(fig3, fullfile(main_save_dir, 'comparison_crosssection.png'));
    saveas(fig3, fullfile(main_save_dir, 'comparison_crosssection.fig'));
end

% Figure 4: Convergence comparison
fig4 = figure('Position', [50, 50, 1200, 500]);
sgtitle('Convergence Comparison', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 2, 1)
semilogy(results.L1.energy(2:end), 'r-', 'LineWidth', 1.5, 'DisplayName', 'L1')
hold on
semilogy(results.L2.energy(2:end), 'g-', 'LineWidth', 1.5, 'DisplayName', 'L2')
semilogy(results.TV.energy(2:end), 'b-', 'LineWidth', 1.5, 'DisplayName', 'TV')
hold off
legend('Location', 'best')
title('Energy', 'FontWeight', 'bold')
xlabel('Iteration'); ylabel('Energy')
grid on

subplot(1, 2, 2)
semilogy(results.L1.resn_set, 'r-', 'LineWidth', 1.5, 'DisplayName', 'L1')
hold on
semilogy(results.L2.resn_set, 'g-', 'LineWidth', 1.5, 'DisplayName', 'L2')
semilogy(results.TV.resn_set, 'b-', 'LineWidth', 1.5, 'DisplayName', 'TV')
hold off
legend('Location', 'best')
title('Residual', 'FontWeight', 'bold')
xlabel('Iteration'); ylabel('||r||^2')
grid on

if save_figures
    saveas(fig4, fullfile(main_save_dir, 'comparison_convergence.png'));
    saveas(fig4, fullfile(main_save_dir, 'comparison_convergence.fig'));
end

% Figure 5: Error maps
fig5 = figure('Position', [50, 50, 1200, 400]);
sgtitle('Absolute Error Maps', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 3, 1)
contourf(x, z, abs(results.L1.c - c_exact), 20); colorbar
title(sprintf('L1 Error (%.4e)', results.L1.final_error), 'FontWeight', 'bold')
axis equal tight

subplot(1, 3, 2)
contourf(x, z, abs(results.L2.c - c_exact), 20); colorbar
title(sprintf('L2 Error (%.4e)', results.L2.final_error), 'FontWeight', 'bold')
axis equal tight

subplot(1, 3, 3)
contourf(x, z, abs(results.TV.c - c_exact), 20); colorbar
title(sprintf('TV Error (%.4e)', results.TV.final_error), 'FontWeight', 'bold')
axis equal tight

if save_figures
    saveas(fig5, fullfile(main_save_dir, 'comparison_error.png'));
    saveas(fig5, fullfile(main_save_dir, 'comparison_error.fig'));
end

%% ========== Save Results ==========
if save_figures
    comparison_data.c_exact = c_exact;
    comparison_data.c0 = c0;
    comparison_data.x = x;
    comparison_data.z = z;
    comparison_data.N = N;
    comparison_data.beta = beta;
    comparison_data.fixed_alpha = fixed_alpha;
    comparison_data.results = results;
    comparison_data.experiment_name = experiment_name;
    
    save(fullfile(main_save_dir, 'comparison_results.mat'), 'comparison_data');
    fprintf('\nAll results saved to: %s\n', main_save_dir);
end

%% ========== Summary Table ==========
fprintf('\n========== Summary ==========\n');
fprintf('%-10s | %-10s | %-12s | %-10s\n', 'Method', 'Iterations', 'Error', 'Time (s)');
fprintf('-----------------------------------------------------\n');
for reg_idx = 1:length(regularization_types)
    reg_type = regularization_types{reg_idx};
    fprintf('%-10s | %-10d | %.6e | %-10.2f\n', ...
            reg_type, results.(reg_type).iterations, ...
            results.(reg_type).final_error, results.(reg_type).time);
end
fprintf('-----------------------------------------------------\n');
