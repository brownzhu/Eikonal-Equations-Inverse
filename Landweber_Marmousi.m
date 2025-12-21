% Landweber iteration for Marmousi model
% Inverse problem for Eikonal equation with L1/L2/TV regularization
%
% Marmousi Model Specifications:
%   - Original grid: 384 x 122 samples
%   - Grid spacing: 24m x 24m
%   - Physical dimensions: 9.192 km (x) x 2.904 km (z)
%
% Why Downsampling?
% -----------------
% The original Marmousi model has 384 x 122 = 46,848 grid points.
% For the inverse problem using Landweber iteration:
%   1. Each iteration requires solving Eikonal equations for ALL source points
%   2. With N_src sources, we solve 2*N_src Eikonal equations per iteration
%   3. Each Eikonal solve has O(N*M*log(N*M)) complexity
%   4. Memory: storing gradients, travel times, etc. scales as O(N*M)
%
% Example computation time estimate:
%   - Full resolution (384x122): ~10-30 seconds per iteration
%   - Downsampled 3x (128x41): ~1-3 seconds per iteration
%   - For 1000 iterations: 3-8 hours vs 15-50 minutes
%
% Recommendation:
%   - Use downsampling (subsample=2 or 3) for algorithm development/testing
%   - Use full resolution for final production runs

clc; clear; close all
tic
format long

%% ========== Experiment Configuration ==========
% Regularization type: 'L1', 'L2', 'TV'
regularization_type = 'TV';

% Whether to save figures
save_figures = true;

% Experiment name
experiment_name = 'marmousi_full';

% Create save directory
save_dir = fullfile(pwd, 'results', [experiment_name '_' regularization_type]);
if save_figures && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% ========== Marmousi Grid Configuration ==========
% Original Marmousi parameters
Nx_orig = 384;      % original samples in x-direction
Nz_orig = 122;      % original samples in z-direction
dx_orig = 24;       % original grid spacing in meters
dz_orig = 24;

% Subsample factor (set to 1 for full resolution)
% subsample = 1: full resolution (384 x 122) - slow but accurate
% subsample = 2: half resolution (192 x 61) - moderate speed
% subsample = 3: third resolution (128 x 41) - fast for testing
subsample = 1;

% Compute grid dimensions
if subsample == 1
    Nx = Nx_orig;
    Nz = Nz_orig;
else
    Nx = ceil(Nx_orig / subsample);
    Nz = ceil(Nz_orig / subsample);
end

% Physical dimensions (in km)
Lx = 9.192;  % total length in x (km)
Lz = 2.904;  % total depth in z (km)

% Grid spacing (in km for consistency with solver)
dx = Lx / (Nx - 1);
dy = Lz / (Nz - 1);  % note: dy is used for z-direction in the solver

% Create coordinate arrays
x = linspace(0, Lx, Nx);
z = linspace(0, Lz, Nz);
[X, Z] = meshgrid(x, z);

fprintf('========== Grid Configuration ==========\n');
fprintf('Subsample factor: %d\n', subsample);
fprintf('Grid size: %d x %d = %d points\n', Nx, Nz, Nx*Nz);
fprintf('Physical size: %.3f km x %.3f km\n', Lx, Lz);
fprintf('Grid spacing: dx=%.4f km, dz=%.4f km\n', dx, dy);

%% ========== Load Marmousi Velocity Model ==========
% Load velocity data from text file
marmousi_file = fullfile(pwd, 'Marmousi4Yuxiao', 'marmousi_smooth.txt');
if exist(marmousi_file, 'file')
    velocity_data = load(marmousi_file);
    fprintf('Loaded Marmousi data: %d x %d\n', size(velocity_data, 1), size(velocity_data, 2));
    
    % Reshape to original grid (122 rows x 384 columns)
    % Note: check if data needs transposing based on file format
    if size(velocity_data, 1) == Nz_orig && size(velocity_data, 2) == Nx_orig
        c_marmousi = velocity_data;
    elseif size(velocity_data, 1) == Nx_orig && size(velocity_data, 2) == Nz_orig
        c_marmousi = velocity_data';
    elseif numel(velocity_data) == Nx_orig * Nz_orig
        % Data is a vector, reshape it
        c_marmousi = reshape(velocity_data, Nz_orig, Nx_orig);
    else
        error('Unexpected Marmousi data dimensions: %d x %d', size(velocity_data, 1), size(velocity_data, 2));
    end
    
    % Downsample if needed
    if subsample > 1
        c_exact = c_marmousi(1:subsample:end, 1:subsample:end);
        % Ensure correct size
        c_exact = c_exact(1:Nz, 1:Nx);
    else
        c_exact = c_marmousi;
    end
    
    fprintf('Velocity model size: %d x %d\n', size(c_exact, 1), size(c_exact, 2));
    fprintf('Velocity range: [%.2f, %.2f] km/s\n', min(c_exact(:)), max(c_exact(:)));
else
    error('Marmousi file not found: %s', marmousi_file);
end

%% ========== Algorithm Parameters ==========
tol = 0.001;
kmax = 1500;

% Landweber parameters
beta = 10.0;              % regularization strength
mu_0 = 0.8*(1 - 1/1.05);  % step size parameter
mu_1 = 600;               % step size upper bound
backCond = mean(c_exact(:));  % background value (use mean velocity)

% Minimum velocity constraint
c_min = min(c_exact(:)) * 0.9;

% Fixed step size (if needed)
use_fixed_alpha = false;
fixed_alpha = 0.01;

%% ========== Source Points Configuration ==========
% Configure source points on the surface (z = 0)
% For seismic, sources are typically on the surface
num_src_x = 8;  % number of sources in x-direction
src_x_indices = round(linspace(20, Nx-20, num_src_x));
src_z_index = 1;  % surface (z = 0)

fixed_pt_list = [];
for i = 1:length(src_x_indices)
    % Format: [0, row_index, col_index]
    fixed_pt_list = [fixed_pt_list; 0, src_z_index, src_x_indices(i)];
end

num_sources = size(fixed_pt_list, 1);
fprintf('Number of sources: %d\n', num_sources);
fprintf('Source x-positions (km): ');
fprintf('%.2f ', x(src_x_indices));
fprintf('\n');

%% ========== Initialization ==========
I = Nz;  % rows (z-direction)
J = Nx;  % columns (x-direction)

niu = 1;
% Initial guess: solve smoothed version with exact boundary
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);
c = c0;

% Initialize xi using regularization inverse
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

% Compute initial c from xi
c = apply_regularization(xi, regularization_type, beta, backCond, c_min, max(I, J));

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
    norm_dual_sq = sum(g(:).^2) * dx * dy;
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
    c = apply_regularization(xi, regularization_type, beta, backCond, c_min, max(I, J));
    
    % Print progress
    if mod(k, 10) == 0
        fprintf('Iter %4d | Energy = %.6e | Residual = %.6e | alpha = %.3e\n', ...
                k, energy(k+1), resn_set(k), alpha_set(k));
    end
end

total_iterations = k;
elapsed_time = toc;

%% ========== Results Summary ==========
final_error = norm(c - c_exact, 'fro') * dx * dy;
relative_error = norm(c - c_exact, 'fro') / norm(c_exact, 'fro');

fprintf('\n========== Results Summary ==========\n');
fprintf('Regularization type: %s\n', regularization_type);
fprintf('Total iterations: %d\n', total_iterations);
fprintf('Final L2 error: %.6e\n', final_error);
fprintf('Relative error: %.4f%%\n', relative_error * 100);
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

%% ========== Figure 1: Velocity Model Comparison ==========
fig1 = figure('Position', [100, 100, 1400, 800]);
sgtitle(sprintf('Marmousi Inversion with %s Regularization', regularization_type), ...
        'FontSize', 14, 'FontWeight', 'bold');

% Consistent colorbar limits
cmin_vel = min([c_exact(:); c(:); c0(:)]);
cmax_vel = max([c_exact(:); c(:); c0(:)]);

subplot(2, 2, 1)
imagesc(x, z, c_exact)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('True Velocity Model', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_vel, cmax_vel])
colorbar
colormap(jet)
axis tight
set(gca, 'YDir', 'reverse')  % depth increases downward

subplot(2, 2, 2)
imagesc(x, z, c0)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('Initial Guess', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_vel, cmax_vel])
colorbar
axis tight
set(gca, 'YDir', 'reverse')

subplot(2, 2, 3)
imagesc(x, z, c)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title(sprintf('Reconstructed (%s)', regularization_type), 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_vel, cmax_vel])
colorbar
axis tight
set(gca, 'YDir', 'reverse')

subplot(2, 2, 4)
imagesc(x, z, abs(c - c_exact))
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('Absolute Error', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis tight
set(gca, 'YDir', 'reverse')

if save_figures
    saveas(fig1, fullfile(save_dir, 'velocity_comparison.png'));
    saveas(fig1, fullfile(save_dir, 'velocity_comparison.fig'));
    fprintf('Saved: velocity_comparison.png\n');
end

%% ========== Figure 2: Convergence ==========
fig2 = figure('Position', [100, 100, 1200, 500]);
sgtitle('Convergence Analysis', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 2, 1)
semilogy(energy(2:end), 'LineWidth', 1.5)
grid on
title('Energy', 'FontWeight', 'bold')
xlabel('Iteration')
ylabel('Energy')

subplot(1, 2, 2)
semilogy(resn_set, 'LineWidth', 1.5)
grid on
title('Residual', 'FontWeight', 'bold')
xlabel('Iteration')
ylabel('||r||^2')

if save_figures
    saveas(fig2, fullfile(save_dir, 'convergence.png'));
    saveas(fig2, fullfile(save_dir, 'convergence.fig'));
    fprintf('Saved: convergence.png\n');
end

%% ========== Figure 3: Depth Profiles ==========
fig3 = figure('Position', [100, 100, 1200, 400]);
sgtitle('Velocity Profiles', 'FontSize', 14, 'FontWeight', 'bold');

% Select profile locations
x_profiles = [Lx*0.25, Lx*0.5, Lx*0.75];

for i = 1:3
    subplot(1, 3, i)
    [~, ix] = min(abs(x - x_profiles(i)));
    plot(c_exact(:, ix), z, 'k-', 'LineWidth', 2, 'DisplayName', 'True')
    hold on
    plot(c0(:, ix), z, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Initial')
    plot(c(:, ix), z, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed')
    hold off
    set(gca, 'YDir', 'reverse')
    xlabel('Velocity (km/s)')
    ylabel('Depth (km)')
    title(sprintf('x = %.2f km', x(ix)))
    legend('Location', 'best')
    grid on
end

if save_figures
    saveas(fig3, fullfile(save_dir, 'depth_profiles.png'));
    saveas(fig3, fullfile(save_dir, 'depth_profiles.fig'));
    fprintf('Saved: depth_profiles.png\n');
end

%% ========== Save Experiment Data ==========
if save_figures
    experiment_data.regularization_type = regularization_type;
    experiment_data.experiment_name = experiment_name;
    experiment_data.Nx = Nx;
    experiment_data.Nz = Nz;
    experiment_data.subsample = subsample;
    experiment_data.Lx = Lx;
    experiment_data.Lz = Lz;
    experiment_data.beta = beta;
    experiment_data.total_iterations = total_iterations;
    experiment_data.final_error = final_error;
    experiment_data.relative_error = relative_error;
    experiment_data.elapsed_time = elapsed_time;
    experiment_data.num_sources = num_sources;
    experiment_data.energy = energy;
    experiment_data.resn_set = resn_set;
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
