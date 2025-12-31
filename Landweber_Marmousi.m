% Landweber iteration for Marmousi model
% Inverse problem for Eikonal equation with L1/L2/TV regularization
%
% Marmousi Model Specifications:
%   - Original grid: 384 x 122 samples
%   - Grid spacing: 24m x 24m
%   - Physical dimensions: 9.192 km (x) x 2.904 km (z)
%
% This version uses fixed grid size (Nx, Nz) without multi-resolution.

clc; clear; close all
tic
format long

%% ========== Experiment Configuration ==========
% Regularization type: 'L1', 'L2', 'TV'
regularization_type = 'TV';
% regularization_type = 'L1';
% regularization_type = 'L2';

% Whether to save figures
save_figures = false;

% Experiment name
experiment_name = 'marmousi_full_noise';

%% ========== Marmousi Grid Configuration ==========
% Original Marmousi parameters (directly use original resolution)
Nx = 384;  % samples in x-direction
Nz = 122;  % samples in z-direction

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
fprintf('Grid size: %d x %d = %d points\n', Nx, Nz, Nx*Nz);
fprintf('Physical size: %.3f km x %.3f km\n', Lx, Lz);
fprintf('Grid spacing: dx=%.4f km, dz=%.4f km\n', dx, dy);

%% ========== Load Marmousi Velocity Model ==========
% Load velocity data from text file (directly use original resolution)
marmousi_file = fullfile(pwd, 'Marmousi4Yuxiao', 'marmousi_smooth.txt');
if exist(marmousi_file, 'file')
    velocity_data = load(marmousi_file);
    fprintf('Loaded Marmousi data: %d x %d\n', size(velocity_data, 1), size(velocity_data, 2));
    
    % Reshape to grid (122 rows x 384 columns)
    if size(velocity_data, 1) == Nz && size(velocity_data, 2) == Nx
        c_exact = velocity_data;
    elseif size(velocity_data, 1) == Nx && size(velocity_data, 2) == Nz
        c_exact = velocity_data';
    elseif numel(velocity_data) == Nx * Nz
        % Data is a vector, reshape it
        c_exact = reshape(velocity_data, Nz, Nx);
    else
        error('Unexpected Marmousi data dimensions: %d x %d', size(velocity_data, 1), size(velocity_data, 2));
    end
    
    fprintf('Velocity model size: %d x %d\n', size(c_exact, 1), size(c_exact, 2));
    fprintf('Velocity range: [%.2f, %.2f] m/s\n', min(c_exact(:)), max(c_exact(:)));
else
    error('Marmousi file not found: %s', marmousi_file);
end
c_exact = c_exact / 1000; % m/s transform to km/s
%% ========== Algorithm Parameters ==========
tol = 1e-12;
kmax = 1200;


p_imp = 0.00;   % 边界点中有 2% 被污染（可调）
A_rel = 0.10;   % 脉冲幅度 = A_rel * (boundary range)（可调）

% Landweber parameters
beta = 0.05;              % regularization strength
duality_r = 1.9;         % duality mapping J_r r value.
mu_0 = 0.8*(1 - 1/1.05);  % step size parameter
mu_1 = 600;               % step size upper bound
backCond = mean(c_exact(:));  % background value (use mean velocity)

% Create save directory (after beta is defined)
p_imp_tag = strrep(sprintf('pImp%.3f', p_imp), '.', 'p');   % e.g. pImp0p020
dr_tag      = strrep(sprintf('dr%.2f', duality_r), '.', 'p');    % e.g. dr2p00
save_dir = fullfile(pwd, 'results', sprintf('%s_%s_%s_%s_beta%.2f', ...
    experiment_name, p_imp_tag, dr_tag, regularization_type, beta));
if save_figures && ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Minimum velocity constraint
c_min = min(c_exact(:)) * 0.9;

% Fixed step size (if needed)
use_fixed_alpha = false;
fixed_alpha = 0.01;

% %% ========== Source Points Configuration ==========
% % Configure source points
% % Format: [0, row_index, col_index]
% 
% num_src_x = 4;  % number of sources in x-direction
% num_src_z = 4;  % number of sources in z-direction
% 
% % Calculate source spacing
% src_x_step = floor((Nx - 2) / (num_src_x + 1));
% src_z_step = floor((Nz - 2) / (num_src_z + 1));
% 
% fixed_pt_list = [];
% for m = 1:num_src_z
%     for n = 1:num_src_x
%         src_z_idx = m * src_z_step;
%         src_x_idx = n * src_x_step;
%         % TravelTime_solver expects format: [val, j, i]
%         % where j = column index (x), i = row index (z)
%         fixed_pt_list = [fixed_pt_list; 0, src_x_idx, src_z_idx];
%     end
% end
% 
% num_sources = size(fixed_pt_list, 1);
% fprintf('Number of sources: %d\n', num_sources);


%% ========== Source Points Configuration (Top & Bottom edges only) ==========
% TravelTime_solver expects: [val, j, i]
%   j = column index (x-direction, 1..Nx)
%   i = row index    (z-direction, 1..Nz)

num_total = 16;            % total number of sources on (top + bottom)
num_src_x = num_total/2;   % sources on top edge
num_src_z = num_total/2;   % sources on bottom edge  (keep same count)

if mod(num_total,2) ~= 0
    error('num_total must be even for equal top/bottom placement.');
end

% ---- grid sizes (consistent with your script) ----
I = Nz;   % rows (z)
J = Nx;   % cols (x)

% ---- choose column indices uniformly (avoid corners by default) ----
% Use interior columns 2..J-1 to avoid putting sources exactly at corners.
% If J is small, this will fall back automatically.
j_min = 2; 
j_max = J-1;
if j_max < j_min
    j_min = 1; j_max = J;   % fallback if grid too narrow
end

j_list = unique(round(linspace(j_min, j_max, num_src_x)), 'stable');

% If uniqueness reduces count (can happen for small J), fill sequentially
while numel(j_list) < num_src_x
    cand = min(j_max, j_list(end) + 1);
    j_list = unique([j_list; cand], 'stable');
    if cand == j_max
        % wrap from j_min upwards if needed
        for jj = j_min:j_max
            j_list = unique([j_list; jj], 'stable');
            if numel(j_list) >= num_src_x, break; end
        end
    end
end
j_list = j_list(1:num_src_x);

% ---- build fixed_pt_list: top i=1, bottom i=I ----
top_list = [zeros(numel(j_list),1), j_list(:), ones(numel(j_list),1)];      % i=1
bot_list = [zeros(numel(j_list),1), j_list(:), I*ones(numel(j_list),1)];   % i=I

fixed_pt_list = [top_list; bot_list];

fprintf('Number of sources: %d (top=%d, bottom=%d)\n', ...
    size(fixed_pt_list,1), size(top_list,1), size(bot_list,1));
num_sources = size(fixed_pt_list, 1);
%% ========== Plot source locations ==========
figure; hold on;
plot(fixed_pt_list(:,2), fixed_pt_list(:,3), 'ro', 'MarkerFaceColor','r');
plot([1 J J 1 1], [1 1 I I 1], 'k-');   % boundary box
axis equal; axis ij; grid on;
xlabel('x index (j)'); ylabel('z index (i)');
title('Source locations on TOP & BOTTOM boundaries', 'FontWeight','bold');
legend({'sources','domain boundary'}, 'Location','best');
hold off;

%% ========== Initialization ==========
I = Nz;  % rows (z-direction)
J = Nx;  % columns (x-direction)

niu = 1;
% Initial guess: solve smoothed version with exact boundary
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);


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


%% ========== observe data (clean) ==========
Tstar_all = zeros(I, J, num_sources);
for p_num = 1:num_sources
    Tstar_all(:,:,p_num) = TravelTime_solver(c_exact, fixed_pt_list(p_num,:), dx, dy, I, J);
end

%% ========== add impulse noise on boundary ==========


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


%% ========== Main Iteration ==========
energy = 1e9;
resn_set = [];
alpha_set = [];
dual_norm_set = [];

fprintf('\n========== Starting Landweber Iteration ==========\n');
fprintf('Regularization: %s, beta = %.2f\n', regularization_type, beta);
resid_min = 1e8;
error_min = 1e8;
c_resi_min = 0;
c_error_min = 0;
for k = 1:kmax
    
    energy_p = 0;
    cstar = 0;
    resn_Lr = 0; 
    
    parfor p_num = 1:size(fixed_pt_list, 1) %parfor
        T = TravelTime_solver(c, fixed_pt_list(p_num, :), dx, dy, I, J);
        T_s = Tstar_noisy(:,:,p_num);
        resn_Lr_ = LrNormBoundary(T, T_s, dx, dy, duality_r)^duality_r;
        resn_Lr = resn_Lr + resn_Lr_;
        energy_p = energy_p + EnergyFun(T, T_s, dx, dy);
        cstar = cstar + cStarSolver(T, T_s, duality_r,  niu, dx, dy, I, J, c);
    end
    
    if energy_p < tol
        fprintf('Converged at iteration %d\n', k);
        break
    end
    resn_Lr = (resn_Lr)^(1/duality_r);
    energy = [energy, energy_p];
    resn_set = [resn_set, resn_Lr];
    
    % Compute dual norm
    g = cstar;  
    norm_dual_sq = sum(g(:).^2) * dx * dy;
    dual_norm_set = [dual_norm_set, norm_dual_sq];
    
    % Compute step size alpha
    if use_fixed_alpha
        alpha = fixed_alpha;
    else
        alpha = min(mu_0 * resn_Lr^(2*(duality_r - 1) ) / max(norm_dual_sq, 1e-12), mu_1) * resn_Lr^(2-duality_r);
    end
    alpha_set = [alpha_set, alpha];
    
    % Update xi
    xi = xi + alpha * cstar;
    
    % Apply Regularization
    c = apply_regularization(xi, regularization_type, beta, backCond, c_min, max(I, J));
    itr_error = norm(c - c_exact, 'fro');
    if resid_min > resn_Lr
        c_resi_min = c;
        resid_min = resn_Lr;
    end

    if error_min > itr_error
        c_error_min = c;
        error_min = itr_error;
    end

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
sgtitle(sprintf('Marmousi Inversion with %s Regularization (\\beta = %.2f)', regularization_type, beta), ...
        'FontSize', 14, 'FontWeight', 'bold');

% Consistent colorbar limits
cmin_vel = min([c_exact(:); c(:); c0(:)]);
cmax_vel = max([c_exact(:); c(:); c0(:)]);

subplot(3, 2, 1)
imagesc(x, z, c_exact)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('True Velocity Model', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_vel, cmax_vel])
colorbar
colormap(jet)
axis tight
axis xy  % z increases upward (normal orientation)

subplot(3, 2, 2)
imagesc(x, z, c0)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('Initial Guess', 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_vel, cmax_vel])
colorbar
axis tight
axis xy

subplot(3, 2, 3)
imagesc(x, z, c)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title(sprintf('Reconstructed (%s, \\beta=%.2f)', regularization_type, beta), 'FontWeight', 'bold', 'FontSize', 12)
caxis([cmin_vel, cmax_vel])
colorbar
axis tight
axis xy


subplot(3, 2, 4)
imagesc(x, z, abs(c - c_exact))
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('Absolute Error', 'FontWeight', 'bold', 'FontSize', 12)
colorbar
axis tight
axis xy

subplot(3, 2, 5)
imagesc(x, z, c_resi_min)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('Minimal Residual Solution: $\|T(c_n)-T^\ast\|$', ...
      'Interpreter','latex','FontWeight','bold','FontSize',12)
caxis([cmin_vel, cmax_vel])
colorbar
axis tight
axis xy

subplot(3, 2, 6)
imagesc(x, z, c_error_min)
xlabel('x (km)', 'FontSize', 12)
ylabel('Depth (km)', 'FontSize', 12)
title('Minimal Error Solution: $ \|c_n-c_{\mathrm{exact}}\|$', ...
      'Interpreter','latex','FontWeight','bold','FontSize',12)
caxis([cmin_vel, cmax_vel])
colorbar
axis tight
axis xy

if save_figures
    saveas(fig1, fullfile(save_dir, 'velocity_comparison.png'));
    saveas(fig1, fullfile(save_dir, 'velocity_comparison.fig'));
    fprintf('Saved: velocity_comparison.png\n');
end

%% ========== Figure 2: Convergence ==========
fig2 = figure('Position', [100, 100, 1200, 500]);
sgtitle(sprintf('Convergence Analysis (%s, \\beta = %.2f)', regularization_type, beta), 'FontSize', 14, 'FontWeight', 'bold');

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
ylabel('||r||')

if save_figures
    saveas(fig2, fullfile(save_dir, 'convergence.png'));
    saveas(fig2, fullfile(save_dir, 'convergence.fig'));
    fprintf('Saved: convergence.png\n');
end

%% ========== Figure 3: Depth Profiles ==========
fig3 = figure('Position', [100, 100, 1200, 400]);
sgtitle(sprintf('Velocity Profiles (%s, \\beta = %.2f)', regularization_type, beta), 'FontSize', 14, 'FontWeight', 'bold');

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
    experiment_data.Nx = Nx;
    experiment_data.Nz = Nz;
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

fprintf('\n========== Experiment Complete ==========\n');
