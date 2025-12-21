% Landweber iteration for Marmousi model with Multi-Resolution Continuation
% 
% Multi-Resolution Strategy:
%   - Start with coarse grid (subsample=3) for fast initial convergence
%   - Interpolate solution to finer grid (subsample=2) and continue
%   - Finally refine to full resolution (subsample=1)
%
% Benefits:
%   1. Much faster overall convergence
%   2. Coarse grid captures global structure quickly
%   3. Fine grid refines local details
%   4. Total time often 2-5x faster than starting at full resolution

clc; clear; close all
total_tic = tic;
format long

%% ========== Experiment Configuration ==========
% Regularization type: 'L1', 'L2', 'TV'
regularization_type = 'L2';

% Whether to save figures
save_figures = true;

% Experiment name
experiment_name = 'marmousi_multires';

% Create save directory
save_dir = fullfile(pwd, 'results', [experiment_name '_' regularization_type]);
if save_figures && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% ========== Multi-Resolution Configuration ==========
% Subsample factors: from coarse to fine
subsample_levels = [3, 2, 1];

% Maximum iterations per level
kmax_per_level = [300, 200, 500];  % more iterations at finer levels

% Tolerance per level (can be relaxed at coarse levels)
tol_per_level = [1e-6, 1e-7, 1e-8];

%% ========== Original Marmousi Parameters ==========
Nx_orig = 384;
Nz_orig = 122;
Lx = 9.192;  % km
Lz = 2.904;  % km

%% ========== Load Full Resolution Marmousi Model ==========
marmousi_file = fullfile(pwd, 'Marmousi4Yuxiao', 'marmousi_smooth.txt');
if exist(marmousi_file, 'file')
    velocity_data = load(marmousi_file);
    fprintf('Loaded Marmousi data: %d x %d\n', size(velocity_data, 1), size(velocity_data, 2));
    
    if size(velocity_data, 1) == Nz_orig && size(velocity_data, 2) == Nx_orig
        c_marmousi_full = velocity_data;
    elseif size(velocity_data, 1) == Nx_orig && size(velocity_data, 2) == Nz_orig
        c_marmousi_full = velocity_data';
    elseif numel(velocity_data) == Nx_orig * Nz_orig
        c_marmousi_full = reshape(velocity_data, Nz_orig, Nx_orig);
    else
        error('Unexpected Marmousi data dimensions');
    end
    c_marmousi_full = c_marmousi_full / 1000;  % m/s to km/s
else
    error('Marmousi file not found: %s', marmousi_file);
end

%% ========== Algorithm Parameters ==========
beta = 10.0;
mu_0 = 0.8*(1 - 1/1.05);
mu_1 = 600;

use_fixed_alpha = false;
fixed_alpha = 0.01;

%% ========== Source Configuration ==========
num_src_x = 8;

%% ========== Multi-Resolution Loop ==========
% Initialize solution (will be interpolated between levels)
c_prev = [];
xi_prev = [];

% Store results for each level
level_results = struct();

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║      MULTI-RESOLUTION LANDWEBER ITERATION                    ║\n');
fprintf('║      Levels: %d → %d → %d                                      ║\n', subsample_levels);
fprintf('╚══════════════════════════════════════════════════════════════╝\n');

for level = 1:length(subsample_levels)
    level_tic = tic;
    subsample = subsample_levels(level);
    kmax = kmax_per_level(level);
    tol = tol_per_level(level);
    
    fprintf('\n');
    fprintf('══════════════════════════════════════════════════════════════\n');
    fprintf('  LEVEL %d/%d: Subsample = %d\n', level, length(subsample_levels), subsample);
    fprintf('══════════════════════════════════════════════════════════════\n');
    
    %% Compute grid for current level
    if subsample == 1
        Nx = Nx_orig;
        Nz = Nz_orig;
    else
        Nx = ceil(Nx_orig / subsample);
        Nz = ceil(Nz_orig / subsample);
    end
    
    dx = Lx / (Nx - 1);
    dy = Lz / (Nz - 1);
    
    x = linspace(0, Lx, Nx);
    z = linspace(0, Lz, Nz);
    [X, Z] = meshgrid(x, z);
    
    fprintf('Grid size: %d x %d = %d points\n', Nx, Nz, Nx*Nz);
    fprintf('Grid spacing: dx=%.4f km, dz=%.4f km\n', dx, dy);
    
    %% Downsample exact solution for current level
    if subsample > 1
        c_exact = c_marmousi_full(1:subsample:end, 1:subsample:end);
        c_exact = c_exact(1:Nz, 1:Nx);
    else
        c_exact = c_marmousi_full;
    end
    c_exact = c_exact/1000;
    backCond = mean(c_exact(:));
    c_min = min(c_exact(:)) * 0.9;
    
    fprintf('Velocity range: [%.3f, %.3f] km/s\n', min(c_exact(:)), max(c_exact(:)));
    
    %% Source points for current level
    src_x_indices = round(linspace(20/subsample, (Nx_orig-20)/subsample, num_src_x));
    src_x_indices = max(2, min(src_x_indices, Nx-1));  % ensure valid indices
    src_z_index = 1;
    
    fixed_pt_list = [];
    for i = 1:length(src_x_indices)
        fixed_pt_list = [fixed_pt_list; 0, src_z_index, src_x_indices(i)];
    end
    
    fprintf('Number of sources: %d\n', size(fixed_pt_list, 1));
    
    %% Initialize or interpolate from previous level
    I = Nz;
    J = Nx;
    niu = 1;
    
    if level == 1
        % First level: use smoothed initial guess
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
        
        fprintf('Initialized from smoothed boundary condition\n');
    else
        % Interpolate from previous level
        [Nz_prev, Nx_prev] = size(c_prev);
        x_prev = linspace(0, Lx, Nx_prev);
        z_prev = linspace(0, Lz, Nz_prev);
        
        % Interpolate c and xi to current grid
        c0 = interp2(x_prev, z_prev', c_prev, X, Z, 'spline');
        xi = interp2(x_prev, z_prev', xi_prev, X, Z, 'spline');
        
        % Handle any NaN from interpolation at boundaries
        c0(isnan(c0)) = backCond;
        xi(isnan(xi)) = backCond;
        
        fprintf('Interpolated from level %d (%dx%d → %dx%d)\n', ...
                level-1, Nx_prev, Nz_prev, Nx, Nz);
    end
    
    c = apply_regularization(xi, regularization_type, beta, backCond, c_min, max(I, J));
    
    %% Landweber iteration for current level
    energy = 1e9;
    resn_set = [];
    alpha_set = [];
    
    fprintf('\nStarting Landweber iteration (max %d, tol=%.1e)...\n', kmax, tol);
    
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
            fprintf('Converged at iteration %d (energy=%.2e < tol=%.2e)\n', k, energy_p, tol);
            break
        end
        
        energy = [energy, energy_p];
        resn_set = [resn_set, resn_power];
        
        % Step size
        g = cstar;
        norm_dual_sq = sum(g(:).^2) * dx * dy;
        
        if use_fixed_alpha
            alpha = fixed_alpha;
        else
            alpha = min(mu_0 * resn_power / max(norm_dual_sq, 1e-12), mu_1);
        end
        alpha_set = [alpha_set, alpha];
        
        % Update
        xi = xi + alpha * cstar;
        c = apply_regularization(xi, regularization_type, beta, backCond, c_min, max(I, J));
        
        if mod(k, 20) == 0
            fprintf('  Iter %4d | Energy = %.6e | Residual = %.6e\n', k, energy_p, resn_power);
        end
    end
    
    level_time = toc(level_tic);
    
    % Store results
    level_results(level).subsample = subsample;
    level_results(level).Nx = Nx;
    level_results(level).Nz = Nz;
    level_results(level).iterations = k;
    level_results(level).final_energy = energy(end);
    level_results(level).time = level_time;
    level_results(level).c = c;
    level_results(level).xi = xi;
    level_results(level).energy = energy;
    level_results(level).resn_set = resn_set;
    
    % Save for next level
    c_prev = c;
    xi_prev = xi;
    
    fprintf('\nLevel %d completed: %d iterations, %.2f seconds\n', level, k, level_time);
    fprintf('Final energy: %.6e\n', energy(end));
end

total_time = toc(total_tic);

%% ========== Final Results ==========
% Use final level results
c_final = level_results(end).c;
x_final = linspace(0, Lx, level_results(end).Nx);
z_final = linspace(0, Lz, level_results(end).Nz);

final_error = norm(c_final - c_exact, 'fro') * dx * dy;
relative_error = norm(c_final - c_exact, 'fro') / norm(c_exact, 'fro');

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║                    FINAL RESULTS                             ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Regularization: %-10s                                   ║\n', regularization_type);
fprintf('║  Total time: %.2f seconds                                  ║\n', total_time);
fprintf('║  Relative error: %.4f%%                                     ║\n', relative_error*100);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Level Summary:                                              ║\n');
for lv = 1:length(subsample_levels)
    fprintf('║    Level %d (sub=%d): %4d iters, %.1fs, E=%.2e          ║\n', ...
            lv, level_results(lv).subsample, level_results(lv).iterations, ...
            level_results(lv).time, level_results(lv).final_energy);
end
fprintf('╚══════════════════════════════════════════════════════════════╝\n');

%% ========== Figure 1: Final Velocity Comparison ==========
fig1 = figure('Position', [100, 100, 1400, 800]);
sgtitle(sprintf('Multi-Resolution Marmousi Inversion (%s)', regularization_type), ...
        'FontSize', 14, 'FontWeight', 'bold');

cmin_vel = min([c_exact(:); c_final(:)]);
cmax_vel = max([c_exact(:); c_final(:)]);

subplot(2, 2, 1)
imagesc(x_final, z_final, c_exact)
xlabel('x (km)'); ylabel('Depth (km)')
title('True Velocity Model')
caxis([cmin_vel, cmax_vel]); colorbar; colormap(jet)
set(gca, 'YDir', 'reverse')

subplot(2, 2, 2)
imagesc(x_final, z_final, c_final)
xlabel('x (km)'); ylabel('Depth (km)')
title(sprintf('Reconstructed (%s)', regularization_type))
caxis([cmin_vel, cmax_vel]); colorbar
set(gca, 'YDir', 'reverse')

subplot(2, 2, 3)
imagesc(x_final, z_final, abs(c_final - c_exact))
xlabel('x (km)'); ylabel('Depth (km)')
title('Absolute Error')
colorbar
set(gca, 'YDir', 'reverse')

subplot(2, 2, 4)
imagesc(x_final, z_final, (c_final - c_exact)./c_exact * 100)
xlabel('x (km)'); ylabel('Depth (km)')
title('Relative Error (%)')
colorbar
set(gca, 'YDir', 'reverse')

if save_figures
    saveas(fig1, fullfile(save_dir, 'velocity_comparison.png'));
    saveas(fig1, fullfile(save_dir, 'velocity_comparison.fig'));
end

%% ========== Figure 2: Convergence across levels ==========
fig2 = figure('Position', [100, 100, 1200, 500]);
sgtitle('Multi-Resolution Convergence', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 2, 1)
colors = {'b', 'g', 'r'};
hold on
offset = 0;
for lv = 1:length(subsample_levels)
    iters = 1:length(level_results(lv).energy)-1;
    semilogy(iters + offset, level_results(lv).energy(2:end), colors{lv}, 'LineWidth', 1.5, ...
             'DisplayName', sprintf('Level %d (sub=%d)', lv, level_results(lv).subsample));
    offset = offset + length(iters);
end
hold off
xlabel('Cumulative Iteration')
ylabel('Energy')
title('Energy vs Iteration')
legend('Location', 'best')
grid on

subplot(1, 2, 2)
bar_data = [level_results.time];
bar(bar_data, 'FaceColor', [0.2 0.6 0.8])
set(gca, 'XTickLabel', arrayfun(@(s) sprintf('sub=%d', s), subsample_levels, 'UniformOutput', false))
xlabel('Resolution Level')
ylabel('Time (seconds)')
title('Computation Time per Level')
grid on

if save_figures
    saveas(fig2, fullfile(save_dir, 'convergence_multires.png'));
    saveas(fig2, fullfile(save_dir, 'convergence_multires.fig'));
end

%% ========== Figure 3: Depth Profiles ==========
fig3 = figure('Position', [100, 100, 1200, 400]);
sgtitle('Velocity Profiles at Different Depths', 'FontSize', 14, 'FontWeight', 'bold');

x_profiles = [Lx*0.25, Lx*0.5, Lx*0.75];

for i = 1:3
    subplot(1, 3, i)
    [~, ix] = min(abs(x_final - x_profiles(i)));
    plot(c_exact(:, ix), z_final, 'k-', 'LineWidth', 2, 'DisplayName', 'True')
    hold on
    plot(c_final(:, ix), z_final, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed')
    hold off
    set(gca, 'YDir', 'reverse')
    xlabel('Velocity (km/s)')
    ylabel('Depth (km)')
    title(sprintf('x = %.2f km', x_final(ix)))
    legend('Location', 'best')
    grid on
end

if save_figures
    saveas(fig3, fullfile(save_dir, 'depth_profiles.png'));
    saveas(fig3, fullfile(save_dir, 'depth_profiles.fig'));
end

%% ========== Save Results ==========
if save_figures
    results.regularization_type = regularization_type;
    results.subsample_levels = subsample_levels;
    results.kmax_per_level = kmax_per_level;
    results.total_time = total_time;
    results.relative_error = relative_error;
    results.level_results = level_results;
    results.c_exact = c_exact;
    results.c_final = c_final;
    results.x = x_final;
    results.z = z_final;
    
    save(fullfile(save_dir, 'multires_results.mat'), 'results');
    fprintf('\nResults saved to: %s\n', save_dir);
end

fprintf('\n========== Multi-Resolution Experiment Complete ==========\n');
