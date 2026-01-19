%% run_duality_r_sweep.m
clc; clear; close all;
format long

% ====== duality mapping r list ======
r_list = [1.2, 1.5, 1.8, 2.0];

% （可选）总结果汇总
summary = struct();

for ir = 1:numel(r_list)
    duality_r = r_list(ir);
    fprintf('\n\n==============================\n');
    fprintf(' Running duality_r = %.2f\n', duality_r);
    fprintf('==============================\n');

    out = run_one_r(duality_r);   % <-- 单次运行
    summary(ir).duality_r       = duality_r;
    summary(ir).save_dir        = out.save_dir;
    summary(ir).final_error     = out.final_error;
    summary(ir).total_iterations= out.total_iterations;
    summary(ir).elapsed_time    = out.elapsed_time;
end

% 保存总汇总
root_dir = fullfile(pwd,'results','duality_r_sweep');
if ~exist(root_dir,'dir'); mkdir(root_dir); end
save(fullfile(root_dir,'summary_all_r.mat'),'summary');
fprintf('\nAll done. Summary saved to: %s\n', fullfile(root_dir,'summary_all_r.mat'));

% 生成对比图（加载每个r的 experiment_data.mat）
plot_compare_all_r(summary);

%% ====== Local functions ======

function out = run_one_r(duality_r)

    tic

    %% ========== Experiment Configuration ==========
    regularization_type = 'L1';  % 你当前用的是 L1
    save_figures = true;         % <-- 为了“全部存下来”，建议强制 true

    experiment_name = 'rectangular_domain_Nes_';

    %% ========== Rectangular Domain Configuration ==========
    x_min = -2; x_max = 2;
    z_min = -1; z_max = 1;

    Nx = 129;
    Nz = 65;

    x = linspace(x_min, x_max, Nx);
    z = linspace(z_min, z_max, Nz);
    [X, Z] = meshgrid(x, z);

    dx = (x_max - x_min) / (Nx - 1);
    dy = (z_max - z_min) / (Nz - 1);

    %% ========== Algorithm Parameters ==========
    tol = 0.001;
    kmax = 1000;

    beta = 0.5;
    Nes_alpha = 3;
    Nesterov  = 0;

    niu = 1;
    % duality_r 由外层传进来

    mu_0 = 0.8*(1 - 1/1.01);
    mu_1 = 600;
    backCond = 1;

    use_fixed_alpha = false;
    fixed_alpha = 0.01;

    %% ========== Save directory (include r) ==========
    save_dir = fullfile(pwd, 'results', sprintf('%s_%s_beta%.2f_r%.2f', ...
        experiment_name, regularization_type, beta, duality_r));
    if save_figures && ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    %% ========== Exact Solution ==========
    I = Nz; J = Nx;
    c_exact = ones(I, J);
    c_min = 0.01;

    mask1 = (X > -0.8 & X < 0.8 & Z > -0.4 & Z < 0.4);
    c_exact(mask1) = c_exact(mask1) + 1.5;

    mask2 = (X > -1.8 & X < -1.2 & Z > -0.6 & Z < 0.6);
    c_exact(mask2) = c_exact(mask2) + 0.8;

    mask3 = (X > 1.2 & X < 1.8 & Z > -0.6 & Z < 0.6);
    c_exact(mask3) = c_exact(mask3) + 0.8;

    c_exact = max(c_exact, c_min);

    %% ========== Source Points ==========
    source_grid_x = 6;
    source_grid_z = 3;

    fixed_pt_list = [];
    source_spacing_i = floor((I - 2) / (source_grid_z + 1));
    source_spacing_j = floor((J - 2) / (source_grid_x + 1));

    for m = 1:source_grid_z
        for n = 1:source_grid_x
            i_pos = 1 + m * source_spacing_i;
            j_pos = 1 + n * source_spacing_j;
            fixed_pt_list = [fixed_pt_list; 0, j_pos, i_pos];
        end
    end
    num_sources = size(fixed_pt_list, 1);

    %% ========== observe data (clean) ==========
    Tstar_all = zeros(I, J, num_sources);
    for p_num = 1:num_sources
        Tstar_all(:,:,p_num) = TravelTime_solver(c_exact, fixed_pt_list(p_num,:), dx, dy, I, J);
    end

    %% ========== add impulse noise on boundary ==========
    p_imp   = 0.05;
    A_rel   = 0.1;
    sigma   = 0.2;
    alpha_n = 0.05;

    Tstar_noisy = Tstar_all;

    rng_seed = 2025;
    rng(rng_seed);

    bmask = false(I,J);
    bmask(1,:) = true; bmask(I,:) = true; bmask(:,1)=true; bmask(:,J)=true;
    bidx = find(bmask);

    for p_num = 1:num_sources
        T0 = Tstar_all(:,:,p_num);

        Tb = T0(bidx);
        range_Tb = max(Tb) - min(Tb) + eps;

        is_imp  = rand(numel(bidx),1) < p_imp;
        imp_idx = bidx(is_imp);
        gauss_idx = bidx(~is_imp);

        A   = A_rel * range_Tb;
        sgn = sign(randn(numel(imp_idx),1));

        T1 = T0;
        T1(imp_idx) = T1(imp_idx) + A .* sgn;

        noise_g = sigma*randn(numel(gauss_idx),1);
        T1(gauss_idx) = T1(gauss_idx) .* (1 + alpha_n * noise_g);

        Tstar_noisy(:,:,p_num) = T1;
    end

    % ====== compute err_level depends on r (as in your code) ======
    err_level = 0;
    for p_num = 1:num_sources
        T_s = Tstar_noisy(:,:,p_num);
        err_level_ = LrNormBoundary(Tstar_all(:,:,p_num), T_s, dx, dy, duality_r)^duality_r;
        err_level = err_level + err_level_;
    end
    err_level = err_level^(1/duality_r);
    disp(err_level);

    %% ========== Initialization ==========
    c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);

    switch regularization_type
        case 'L2'
            xi = c0;
        case 'L1'
            q0 = sign(c0 - backCond);
            xi = c0 + beta * q0;
        case 'TV'
            w1 = zeros(size(c0)); w2 = zeros(size(c0));
            [~, w1, w2] = TV_PDHG_host(w1, w2, c0, beta, 100, 1e-6);
            div_w = ([w1(:,1), w1(:,2:end)-w1(:,1:end-1)] + ...
                     [w2(1,:); w2(2:end,:)-w2(1:end-1,:)]);
            q0 = -div_w;
            xi = c0 + beta * q0;
    end

    c  = apply_regularization(xi, regularization_type, beta, backCond, c_min, []);
    c0 = c;
    xi_n = xi;
    zeta = 0;

    %% ========== Main Iteration ==========
    energy = [];
    resn_set = [];
    alpha_set = [];
    dual_norm_set = [];
    c_error_set = [];

    resn_Lr = 100;
    k = 1;

    while resn_Lr >= err_level*1.1 && k <= kmax

        energy_p = 0;
        cstar = 0;
        resn_Lr = 0;
        
        % For parfor: need to accumulate properly
        resn_Lr_total = 0;
        energy_p_total = 0;
        cstar_total = 0;

        if Nesterov
            Nes_lambda = (k)/(k + Nes_alpha);
        else
            Nes_lambda = 0;
        end
        zeta = xi + Nes_lambda * (xi - xi_n);

        parfor p_num = 1:num_sources
            T   = TravelTime_solver(c, fixed_pt_list(p_num, :), dx, dy, I, J);
            T_s = Tstar_noisy(:,:,p_num);

            resn_Lr_  = LrNormBoundary(T, T_s, dx, dy, duality_r)^duality_r;
            resn_Lr_total   = resn_Lr_total + resn_Lr_;
            energy_p_total  = energy_p_total + EnergyFun(T, T_s, dx, dy);
            cstar_total     = cstar_total + cStarSolver(T, T_s, duality_r, niu, dx, dy, I, J, c);
        end
        
        energy_p = energy_p_total;
        resn_Lr = resn_Lr_total;
        cstar = cstar_total;

        if energy_p < tol
            fprintf('Converged at iteration %d\n', k);
            break
        end

        resn_Lr = (resn_Lr)^(1/duality_r);
        energy = [energy, energy_p];
        resn_set = [resn_set, resn_Lr];

        g = cstar;
        norm_dual_sq = sum(g(:).^2) * dx * dy;
        dual_norm_set = [dual_norm_set, norm_dual_sq];

        if use_fixed_alpha
            alpha = fixed_alpha;
        else
            alpha = min(mu_0 * resn_Lr^(2*(duality_r - 1)) / max(norm_dual_sq, 1e-12), mu_1) ...
                    * resn_Lr^(2-duality_r);
        end
        alpha_set = [alpha_set, alpha];

        xi_n = xi;
        xi = zeta + alpha * cstar;

        c = apply_regularization(xi, regularization_type, beta, backCond, c_min, []);
        c_error = norm(c - c_exact, 'fro')*sqrt(dx*dy);
        c_error_set = [c_error_set, c_error];

        if mod(k,10)==0
            fprintf('r=%.2f | Iter %4d | Energy=%.3e | Res=%.3e | Err=%.3e | alpha=%.3e\n', ...
                duality_r, k, energy_p, resn_set(k), c_error_set(k), alpha_set(k));
        end

        k = k + 1;
    end

    total_iterations = k;
    elapsed_time = toc;
    final_error = norm(c - c_exact, 'fro') * sqrt(dx * dy);

    %% ========== Save experiment_data (always) ==========
    experiment_data = struct();
    experiment_data.regularization_type = regularization_type;
    experiment_data.experiment_name = experiment_name;
    experiment_data.domain = [x_min, x_max, z_min, z_max];
    experiment_data.Nx = Nx; experiment_data.Nz = Nz;
    experiment_data.I = I; experiment_data.J = J;
    experiment_data.dx = dx; experiment_data.dy = dy;

    experiment_data.beta = beta;
    experiment_data.tol  = tol;
    experiment_data.kmax = kmax;

    experiment_data.total_iterations = total_iterations;
    experiment_data.final_error = final_error;
    experiment_data.elapsed_time = elapsed_time;

    experiment_data.num_sources = num_sources;

    experiment_data.energy = energy;
    experiment_data.resn_set = resn_set;
    experiment_data.alpha_set = alpha_set;
    experiment_data.dual_norm_set = dual_norm_set;
    experiment_data.c_error_set = c_error_set;

    experiment_data.c_exact = c_exact;
    experiment_data.c_solution = c;
    experiment_data.c0 = c0;
    experiment_data.x = x;
    experiment_data.z = z;

    experiment_data.noise.type = 'impulse_boundary_additive';
    experiment_data.noise.p_imp = p_imp;
    experiment_data.noise.A_rel = A_rel;
    experiment_data.noise.rng_seed = rng_seed;
    experiment_data.noise.boundary_only = true;

    experiment_data.duality_r = duality_r;
    experiment_data.err_level = err_level;

    if ~exist(save_dir,'dir'); mkdir(save_dir); end
    save(fullfile(save_dir,'experiment_data.mat'),'experiment_data');
    fprintf('Saved: %s\n', fullfile(save_dir,'experiment_data.mat'));

    % 绘制精细学术风格的结果图表
    if save_figures && ~isempty(energy) && ~isempty(resn_set) && ~isempty(c_error_set)
        % 设置全局字体属性
        fontname_main = 'Times New Roman';
        fontsize_main = 11;
        fontsize_label = 12;
        fontsize_title = 13;
        
        % 定义颜色方案（学术风格）
        color_energy = [0.2, 0.4, 0.8];   % 深蓝
        color_residual = [0.8, 0.2, 0.2]; % 深红
        color_error = [0.2, 0.7, 0.3];    % 深绿
        
        fig = figure('Position',[100,100,1600,450]);
        
        % Energy curve
        ax1 = subplot(1,3,1);
        if length(energy) > 1
            semilogy(energy(2:end), 'Color', color_energy, 'LineWidth', 1.8, 'LineStyle', '-');
        else
            semilogy(energy, 'Color', color_energy, 'LineWidth', 1.8, 'LineStyle', '-');
        end
        hold on;
        grid on;
        set(ax1, 'GridLineStyle', '--', 'GridAlpha', 0.3);
        xlabel('Iteration', 'FontName', fontname_main, 'FontSize', fontsize_label);
        ylabel('Energy $E$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
        title('Energy', 'FontName', fontname_main, 'FontSize', fontsize_title, 'FontWeight', 'bold');
        set(ax1, 'FontName', fontname_main, 'FontSize', fontsize_main);
        
        % Residual curve
        ax2 = subplot(1,3,2);
        semilogy(resn_set, 'Color', color_residual, 'LineWidth', 1.8, 'LineStyle', '-');
        hold on;
        grid on;
        set(ax2, 'GridLineStyle', '--', 'GridAlpha', 0.3);
        xlabel('Iteration', 'FontName', fontname_main, 'FontSize', fontsize_label);
        ylabel('Residual $\|\mathbf{r}\|_r$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
        title('Residual', 'FontName', fontname_main, 'FontSize', fontsize_title, 'FontWeight', 'bold');
        set(ax2, 'FontName', fontname_main, 'FontSize', fontsize_main);
        
        % c-error curve
        ax3 = subplot(1,3,3);
        semilogy(c_error_set, 'Color', color_error, 'LineWidth', 1.8, 'LineStyle', '-');
        hold on;
        grid on;
        set(ax3, 'GridLineStyle', '--', 'GridAlpha', 0.3);
        xlabel('Iteration', 'FontName', fontname_main, 'FontSize', fontsize_label);
        ylabel('Error $\|c - c_{\mathrm{exact}}\|_2$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
        title('Solution Error', 'FontName', fontname_main, 'FontSize', fontsize_title, 'FontWeight', 'bold');
        set(ax3, 'FontName', fontname_main, 'FontSize', fontsize_main);
        
        % 总标题
        sgtitle(sprintf('Convergence Analysis: $r = %.2f$', duality_r), ...
            'FontName', fontname_main, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');
        
        % 保存高分辨率图片
        set(fig, 'Renderer', 'painters');
        set(fig, 'PaperUnits', 'centimeters');
        set(fig, 'PaperSize', [40, 11.25]);
        set(fig, 'PaperPosition', [0, 0, 40, 11.25]);
        print(fig, fullfile(save_dir, 'convergence_curves.pdf'), '-dpdf', '-r300', '-fillpage');
        saveas(fig, fullfile(save_dir, 'convergence_curves.png'), 'png');
        close(fig);
    elseif save_figures
        warning('Insufficient data to generate convergence plots for r=%.2f', duality_r);
    end

    out.save_dir = save_dir;
    out.final_error = final_error;
    out.total_iterations = total_iterations;
    out.elapsed_time = elapsed_time;
end

function plot_compare_all_r(summary)
    % 生成学术风格的多参数对比图
    nR = numel(summary);
    data = cell(nR,1);
    for i = 1:nR
        S = load(fullfile(summary(i).save_dir,'experiment_data.mat'));
        data{i} = S.experiment_data;
    end

    % 字体和风格设置
    fontname_main = 'Times New Roman';
    fontsize_main = 11;
    fontsize_label = 12;
    fontsize_title = 13;
    fontsize_legend = 11;
    
    % 颜色方案（多种颜色用于区分不同的r值）
    colors = {
        [0.1, 0.3, 0.8],   % 深蓝
        [0.8, 0.2, 0.2],   % 深红
        [0.2, 0.7, 0.3],   % 深绿
        [0.8, 0.6, 0.1],   % 金黄
        [0.6, 0.2, 0.8],   % 深紫
        [0.2, 0.7, 0.8],   % 青色
    };
    
    % 线条样式
    line_styles = {'-', '--', '-.', ':'};
    
    root_dir = fullfile(pwd,'results','duality_r_sweep');
    if ~exist(root_dir,'dir'); mkdir(root_dir); end

    %% 1) Residual 对比图
    fig1 = figure('Position',[100,100,1200,500], 'Units', 'pixels');
    ax1 = axes('Parent', fig1);
    hold(ax1, 'on');
    grid(ax1, 'on');
    set(ax1, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    
    for i = 1:nR
        color_idx = mod(i-1, numel(colors)) + 1;
        line_idx = mod(i-1, numel(line_styles)) + 1;
        
        % 计算合理的 marker indices
        data_len = length(data{i}.resn_set);
        if data_len > 1
            marker_step = max(1, floor(data_len / 8));
            marker_indices = 1:marker_step:data_len;
        else
            marker_indices = 1;
        end
        
        semilogy(ax1, data{i}.resn_set, 'LineWidth', 2, ...
            'Color', colors{color_idx}, 'LineStyle', line_styles{line_idx}, ...
            'DisplayName', sprintf('$r = %.2f$', data{i}.duality_r), 'Marker', 'o', 'MarkerSize', 3, 'MarkerIndices', marker_indices);
    end
    
    xlabel('Iteration $k$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
    ylabel('Residual $\|r_k\|_r$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
    title('Comparison of Residual for Different $r$', 'FontName', fontname_main, 'FontSize', fontsize_title, 'FontWeight', 'bold', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', fontsize_legend, 'FontName', fontname_main);
    set(ax1, 'FontName', fontname_main, 'FontSize', fontsize_main);
    
    set(fig1, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [30, 12.5], 'PaperPosition', [0, 0, 30, 12.5]);
    print(fig1, fullfile(root_dir,'compare_residual.pdf'), '-dpdf', '-r300', '-fillpage');
    saveas(fig1, fullfile(root_dir,'compare_residual.png'), 'png');
    close(fig1);

    %% 2) c-error 对比图
    fig2 = figure('Position',[100,100,1200,500], 'Units', 'pixels');
    ax2 = axes('Parent', fig2);
    hold(ax2, 'on');
    grid(ax2, 'on');
    set(ax2, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    
    for i = 1:nR
        color_idx = mod(i-1, numel(colors)) + 1;
        line_idx = mod(i-1, numel(line_styles)) + 1;
        
        % 计算合理的 marker indices
        data_len = length(data{i}.c_error_set);
        if data_len > 1
            marker_step = max(1, floor(data_len / 8));
            marker_indices = 1:marker_step:data_len;
        else
            marker_indices = 1;
        end
        
        semilogy(ax2, data{i}.c_error_set, 'LineWidth', 2, ...
            'Color', colors{color_idx}, 'LineStyle', line_styles{line_idx}, ...
            'DisplayName', sprintf('$r = %.2f$', data{i}.duality_r), 'Marker', 's', 'MarkerSize', 3, 'MarkerIndices', marker_indices);
    end
    
    xlabel('Iteration $k$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
    ylabel('Solution Error $\|c - c_{\mathrm{exact}}\|_2$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
    title('Comparison of Solution Error for Different $r$', 'FontName', fontname_main, 'FontSize', fontsize_title, 'FontWeight', 'bold', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', fontsize_legend, 'FontName', fontname_main);
    set(ax2, 'FontName', fontname_main, 'FontSize', fontsize_main);
    
    set(fig2, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [30, 12.5], 'PaperPosition', [0, 0, 30, 12.5]);
    print(fig2, fullfile(root_dir,'compare_c_error.pdf'), '-dpdf', '-r300', '-fillpage');
    saveas(fig2, fullfile(root_dir,'compare_c_error.png'), 'png');
    close(fig2);

    %% 3) Step Size α 对比图
    fig3 = figure('Position',[100,100,1200,500], 'Units', 'pixels');
    ax3 = axes('Parent', fig3);
    hold(ax3, 'on');
    grid(ax3, 'on');
    set(ax3, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    
    for i = 1:nR
        color_idx = mod(i-1, numel(colors)) + 1;
        line_idx = mod(i-1, numel(line_styles)) + 1;
        
        % 计算合理的 marker indices
        data_len = length(data{i}.alpha_set);
        if data_len > 1
            marker_step = max(1, floor(data_len / 8));
            marker_indices = 1:marker_step:data_len;
        else
            marker_indices = 1;
        end
        
        semilogy(ax3, data{i}.alpha_set, 'LineWidth', 2, ...
            'Color', colors{color_idx}, 'LineStyle', line_styles{line_idx}, ...
            'DisplayName', sprintf('$r = %.2f$', data{i}.duality_r), 'Marker', '^', 'MarkerSize', 3, 'MarkerIndices', marker_indices);
    end
    
    xlabel('Iteration $k$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
    ylabel('Step Size $\alpha_k$', 'FontName', fontname_main, 'FontSize', fontsize_label, 'Interpreter', 'latex');
    title('Comparison of Step Size for Different $r$', 'FontName', fontname_main, 'FontSize', fontsize_title, 'FontWeight', 'bold', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', fontsize_legend, 'FontName', fontname_main);
    set(ax3, 'FontName', fontname_main, 'FontSize', fontsize_main);
    
    set(fig3, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [30, 12.5], 'PaperPosition', [0, 0, 30, 12.5]);
    print(fig3, fullfile(root_dir,'compare_alpha.pdf'), '-dpdf', '-r300', '-fillpage');
    saveas(fig3, fullfile(root_dir,'compare_alpha.png'), 'png');
    close(fig3);

    fprintf('\n========== Compare figures saved to: %s ==========\n', root_dir);
    fprintf('  - compare_residual.pdf/png\n');
    fprintf('  - compare_c_error.pdf/png\n');
    fprintf('  - compare_alpha.pdf/png\n');
    
    %% 4) 10个子图的综合对比：4个数值结果 + 4个误差 + 2个cross-section
    plot_10subplot_comparison(data, root_dir);
end

function plot_10subplot_comparison(data, root_dir)
    % 生成包含12个子图的综合对比图 + 每个子图单独存储
    % 布局（3行4列）：
    % 第一行：精确解 + r=1.2 + r=1.5 + r=1.8
    % 第二行：r=2.0 + 误差r=1.2 + 误差r=1.5 + 误差r=1.8
    % 第三行：误差r=2.0 + cross-section z=0 + cross-section x=0 + 空白
    
    nR = numel(data);
    if nR < 4
        warning('Need at least 4 r values for comprehensive comparison. Got %d.', nR);
        return;
    end
    
    % 提取所有4个r值
    r_indices = 1:4;
    c_exact = data{1}.c_exact;
    x = data{1}.x;
    z = data{1}.z;
    I = data{1}.I;
    J = data{1}.J;
    dx = data{1}.dx;
    dy = data{1}.dy;
    
    % 创建大图（3行4列）
    fig = figure('Position', [50, 50, 1600, 1200], 'Units', 'pixels');
    set(fig, 'Color', 'white');
    
    % 设置字体
    fontname_main = 'Times New Roman';
    fontsize_label = 11;
    fontsize_title = 12;
    
    % 计算全局colorbar范围（对数值图）
    c_min_all = min(c_exact(:));
    c_max_all = max(c_exact(:));
    for i = r_indices
        c_sol = data{i}.c_solution;
        c_min_all = min(c_min_all, min(c_sol(:)));
        c_max_all = max(c_max_all, max(c_sol(:)));
    end
    
    % 误差的colorbar范围
    err_max_all = 0;
    for i = r_indices
        err = abs(data{i}.c_solution - c_exact);
        err_max_all = max(err_max_all, max(err(:)));
    end
    
    % ========== 第一行：1个精确解 + 3个反演解 ==========
    % 子图1：精确解
    ax1 = subplot(3, 4, 1);
    imagesc(x, z, c_exact, [c_min_all, c_max_all]);
    set(ax1, 'YDir', 'normal');
    colorbar(ax1, 'FontSize', fontsize_label-1);
    title('$c_{\mathrm{exact}}$', 'Interpreter', 'latex', 'FontSize', fontsize_title, 'FontWeight', 'bold');
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    set(ax1, 'FontName', fontname_main, 'FontSize', fontsize_label-1);
    
    % 子图2-4：前3个r的反演解
    for idx = 1:3
        i = r_indices(idx);
        ax = subplot(3, 4, 1 + idx);
        c_sol = data{i}.c_solution;
        imagesc(x, z, c_sol, [c_min_all, c_max_all]);
        set(ax, 'YDir', 'normal');
        colorbar(ax, 'FontSize', fontsize_label-1);
        title(sprintf('$c_{r=%.2f}$', data{i}.duality_r), 'Interpreter', 'latex', ...
            'FontSize', fontsize_title, 'FontWeight', 'bold');
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        set(ax, 'FontName', fontname_main, 'FontSize', fontsize_label-1);
    end
    
    % ========== 第二行：1个反演解(r=2.0) + 3个误差 ==========
    % 子图5：第4个r的反演解
    ax5 = subplot(3, 4, 5);
    c_sol = data{4}.c_solution;
    imagesc(x, z, c_sol, [c_min_all, c_max_all]);
    set(ax5, 'YDir', 'normal');
    colorbar(ax5, 'FontSize', fontsize_label-1);
    title(sprintf('$c_{r=%.2f}$', data{4}.duality_r), 'Interpreter', 'latex', ...
        'FontSize', fontsize_title, 'FontWeight', 'bold');
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    set(ax5, 'FontName', fontname_main, 'FontSize', fontsize_label-1);
    
    % 子图6-8：前3个r的误差
    for idx = 1:3
        i = r_indices(idx);
        ax = subplot(3, 4, 5 + idx);
        c_sol = data{i}.c_solution;
        err = abs(c_sol - c_exact);
        imagesc(x, z, err, [0, err_max_all]);
        set(ax, 'YDir', 'normal');
        colorbar(ax, 'FontSize', fontsize_label-1);
        title(sprintf('Error $r=%.2f$', data{i}.duality_r), 'Interpreter', 'latex', ...
            'FontSize', fontsize_title, 'FontWeight', 'bold');
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        set(ax, 'FontName', fontname_main, 'FontSize', fontsize_label-1);
    end
    
    % ========== 第三行：1个误差(r=2.0) + 2个cross-section + 1个空白 ==========
    % 子图9：第4个r的误差
    ax9 = subplot(3, 4, 9);
    c_sol = data{4}.c_solution;
    err = abs(c_sol - c_exact);
    imagesc(x, z, err, [0, err_max_all]);
    set(ax9, 'YDir', 'normal');
    colorbar(ax9, 'FontSize', fontsize_label-1);
    title(sprintf('Error $r=%.2f$', data{4}.duality_r), 'Interpreter', 'latex', ...
        'FontSize', fontsize_title, 'FontWeight', 'bold');
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    set(ax9, 'FontName', fontname_main, 'FontSize', fontsize_label-1);
    
    % 子图10：Cross-section at z=0
    ax10 = subplot(3, 4, 10);
    hold(ax10, 'on');
    mid_i = ceil(I / 2);
    plot(x, c_exact(mid_i, :), '-', 'LineWidth', 2.5, 'DisplayName', '$c_{\mathrm{exact}}$', ...
        'Color', [0, 0, 0]);
    
    colors_cross = {[0.1, 0.3, 0.8], [0.8, 0.2, 0.2], [0.2, 0.7, 0.3], [0.8, 0.6, 0.1]};
    line_styles_cross = {'--', '-.', ':', '-.'};
    for idx = 1:4
        i = r_indices(idx);
        c_sol = data{i}.c_solution;
        plot(x, c_sol(mid_i, :), line_styles_cross{idx}, 'LineWidth', 2, ...
            'DisplayName', sprintf('$r=%.2f$', data{i}.duality_r), ...
            'Color', colors_cross{idx});
    end
    
    hold(ax10, 'off');
    grid(ax10, 'on');
    set(ax10, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$c$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    title('Cross-section at $z=0$', 'Interpreter', 'latex', 'FontSize', fontsize_title, 'FontWeight', 'bold');
    legend('Location', 'northeastoutside', 'Interpreter', 'latex', 'FontSize', fontsize_label-1, ...
        'FontName', fontname_main, 'NumColumns', 1);
    set(ax10, 'FontName', fontname_main, 'FontSize', fontsize_label);
    
    % 子图11：Cross-section at x=0
    ax11 = subplot(3, 4, 11);
    hold(ax11, 'on');
    mid_j = ceil(J / 2);
    plot(z, c_exact(:, mid_j), '-', 'LineWidth', 2.5, 'DisplayName', '$c_{\mathrm{exact}}$', ...
        'Color', [0, 0, 0]);
    
    for idx = 1:4
        i = r_indices(idx);
        c_sol = data{i}.c_solution;
        plot(z, c_sol(:, mid_j), line_styles_cross{idx}, 'LineWidth', 2, ...
            'DisplayName', sprintf('$r=%.2f$', data{i}.duality_r), ...
            'Color', colors_cross{idx});
    end
    
    hold(ax11, 'off');
    grid(ax11, 'on');
    set(ax11, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    xlabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$c$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    title('Cross-section at $x=0$', 'Interpreter', 'latex', 'FontSize', fontsize_title, 'FontWeight', 'bold');
    legend('Location', 'northeastoutside', 'Interpreter', 'latex', 'FontSize', fontsize_label-1, ...
        'FontName', fontname_main, 'NumColumns', 1);
    set(ax11, 'FontName', fontname_main, 'FontSize', fontsize_label);
    
    % 子图12：空白
    ax12 = subplot(3, 4, 12);
    axis off;
    
    % 总标题
    sgtitle(sprintf('Comprehensive Comparison: Solutions, Errors & Cross-sections (4 r-values)'), ...
        'FontSize', 14, 'FontWeight', 'bold', 'FontName', fontname_main);
    
    % 保存综合图
    set(fig, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [40, 30], ...
        'PaperPosition', [0, 0, 40, 30]);
    drawnow;
    print(fig, fullfile(root_dir, 'comprehensive_full_comparison.pdf'), '-dpdf', '-r300', '-fillpage');
    print(fig, fullfile(root_dir, 'comprehensive_full_comparison.eps'), '-depsc2', '-r300');
    print(fig, fullfile(root_dir, 'comprehensive_full_comparison.png'), '-dpng', '-r300');
    close(fig);
    
    fprintf('✓ Saved comprehensive 12-subplot comparison to: %s\n', root_dir);
    
    % ========== 单独存储每个子图 ==========
    save_individual_subplots(data, c_exact, x, z, I, J, c_min_all, c_max_all, err_max_all, root_dir, fontname_main, fontsize_label, fontsize_title);
end

function save_individual_subplots(data, c_exact, x, z, I, J, c_min_all, c_max_all, err_max_all, root_dir, fontname_main, fontsize_label, fontsize_title)
    % 创建子目录用于存储单个子图
    subdir = fullfile(root_dir, 'individual_subplots');
    if ~exist(subdir, 'dir')
        mkdir(subdir);
    end
    
    nR = min(4, numel(data));
    
    % ========== 保存精确解 ==========
    fig = figure('Position', [50, 50, 600, 550], 'Units', 'pixels', 'Visible', 'off');
    imagesc(x, z, c_exact, [c_min_all, c_max_all]);
    set(gca, 'YDir', 'normal');
    colorbar('FontSize', fontsize_label-1);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    title('$c_{\mathrm{exact}}$', 'Interpreter', 'latex', 'FontSize', fontsize_title+1, 'FontWeight', 'bold');
    set(gca, 'FontName', fontname_main, 'FontSize', fontsize_label);
    
    set(fig, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [15, 13.5], 'PaperPosition', [0, 0, 15, 13.5]);
    drawnow;
    print(fig, fullfile(subdir, '01_exact_solution.pdf'), '-dpdf', '-r300', '-fillpage');
    print(fig, fullfile(subdir, '01_exact_solution.eps'), '-depsc2', '-r300');
    print(fig, fullfile(subdir, '01_exact_solution.png'), '-dpng', '-r300');
    close(fig);
    fprintf('  → 01_exact_solution\n');
    
    % ========== 保存4个反演解 ==========
    for idx = 1:nR
        i = idx;
        fig = figure('Position', [50, 50, 600, 550], 'Units', 'pixels', 'Visible', 'off');
        c_sol = data{i}.c_solution;
        imagesc(x, z, c_sol, [c_min_all, c_max_all]);
        set(gca, 'YDir', 'normal');
        colorbar('FontSize', fontsize_label-1);
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        title(sprintf('$c_{r=%.2f}$', data{i}.duality_r), 'Interpreter', 'latex', ...
            'FontSize', fontsize_title+1, 'FontWeight', 'bold');
        set(gca, 'FontName', fontname_main, 'FontSize', fontsize_label);
        
        set(fig, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [15, 13.5], 'PaperPosition', [0, 0, 15, 13.5]);
        filename = sprintf('0%d_solution_r_%.2f.pdf', idx+1, data{i}.duality_r);
        print(fig, fullfile(subdir, filename), '-dpdf', '-r300', '-fillpage');
        filename_eps = sprintf('0%d_solution_r_%.2f.eps', idx+1, data{i}.duality_r);
        print(fig, fullfile(subdir, filename_eps), '-depsc2', '-r300');
        filename_png = sprintf('0%d_solution_r_%.2f.png', idx+1, data{i}.duality_r);
        print(fig, fullfile(subdir, filename_png), '-dpng', '-r300');
        close(fig);
        fprintf('  → 0%d_solution_r_%.2f\n', idx+1, data{i}.duality_r);
    end
    
    % ========== 保存4个误差图 ==========
    for idx = 1:nR
        i = idx;
        fig = figure('Position', [50, 50, 600, 550], 'Units', 'pixels', 'Visible', 'off');
        c_sol = data{i}.c_solution;
        err = abs(c_sol - c_exact);
        imagesc(x, z, err, [0, err_max_all]);
        set(gca, 'YDir', 'normal');
        colorbar('FontSize', fontsize_label-1);
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        ylabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
        title_str = ['Error $|c_{r=', sprintf('%.2f', data{i}.duality_r), '} - c_{\mathrm{exact}}|$'];
        title(title_str, 'Interpreter', 'latex', ...
            'FontSize', fontsize_title+1, 'FontWeight', 'bold');
        set(gca, 'FontName', fontname_main, 'FontSize', fontsize_label);
        
        set(fig, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [15, 13.5], 'PaperPosition', [0, 0, 15, 13.5]);
        filename = sprintf('0%d_error_r_%.2f.pdf', idx+5, data{i}.duality_r);
        print(fig, fullfile(subdir, filename), '-dpdf', '-r300', '-fillpage');
        filename_eps = sprintf('0%d_error_r_%.2f.eps', idx+5, data{i}.duality_r);
        print(fig, fullfile(subdir, filename_eps), '-depsc2', '-r300');
        filename_png = sprintf('0%d_error_r_%.2f.png', idx+5, data{i}.duality_r);
        print(fig, fullfile(subdir, filename_png), '-dpng', '-r300');
        close(fig);
        fprintf('  → 0%d_error_r_%.2f\n', idx+5, data{i}.duality_r);
    end
    
    % ========== 保存cross-section图 ==========
    % Cross-section at z=0
    fig = figure('Position', [50, 50, 700, 500], 'Units', 'pixels', 'Visible', 'off');
    hold on;
    mid_i = ceil(I / 2);
    plot(x, c_exact(mid_i, :), '-', 'LineWidth', 2.5, 'DisplayName', '$c_{\mathrm{exact}}$', 'Color', [0, 0, 0]);
    
    colors_cross = {[0.1, 0.3, 0.8], [0.8, 0.2, 0.2], [0.2, 0.7, 0.3], [0.8, 0.6, 0.1]};
    line_styles_cross = {'--', '-.', ':', '-.'};
    for idx = 1:nR
        i = idx;
        c_sol = data{i}.c_solution;
        plot(x, c_sol(mid_i, :), line_styles_cross{idx}, 'LineWidth', 2, ...
            'DisplayName', sprintf('$r=%.2f$', data{i}.duality_r), 'Color', colors_cross{idx});
    end
    
    hold off;
    grid on;
    set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$c$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    title('Cross-section at $z=0$ (All $r$ values)', 'Interpreter', 'latex', ...
        'FontSize', fontsize_title+1, 'FontWeight', 'bold');
    legend('Location', 'northeastoutside', 'Interpreter', 'latex', 'FontSize', fontsize_label, 'FontName', fontname_main, 'NumColumns', 1);
    set(gca, 'FontName', fontname_main, 'FontSize', fontsize_label);
    
    set(fig, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [17.5, 12.5], 'PaperPosition', [0, 0, 17.5, 12.5]);
    drawnow;
    print(fig, fullfile(subdir, '09_crosssection_z0.pdf'), '-dpdf', '-r300', '-fillpage');
    print(fig, fullfile(subdir, '09_crosssection_z0.eps'), '-depsc2', '-r300');
    print(fig, fullfile(subdir, '09_crosssection_z0.png'), '-dpng', '-r300');
    close(fig);
    fprintf('  → 09_crosssection_z0\n');
    
    % Cross-section at x=0
    fig = figure('Position', [50, 50, 700, 500], 'Units', 'pixels', 'Visible', 'off');
    hold on;
    mid_j = ceil(J / 2);
    plot(z, c_exact(:, mid_j), '-', 'LineWidth', 2.5, 'DisplayName', '$c_{\mathrm{exact}}$', 'Color', [0, 0, 0]);
    
    for idx = 1:nR
        i = idx;
        c_sol = data{i}.c_solution;
        plot(z, c_sol(:, mid_j), line_styles_cross{idx}, 'LineWidth', 2, ...
            'DisplayName', sprintf('$r=%.2f$', data{i}.duality_r), 'Color', colors_cross{idx});
    end
    
    hold off;
    grid on;
    set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
    xlabel('$z$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    ylabel('$c$', 'Interpreter', 'latex', 'FontSize', fontsize_label);
    title('Cross-section at $x=0$ (All $r$ values)', 'Interpreter', 'latex', ...
        'FontSize', fontsize_title+1, 'FontWeight', 'bold');
    legend('Location', 'northeastoutside', 'Interpreter', 'latex', 'FontSize', fontsize_label, 'FontName', fontname_main, 'NumColumns', 1);
    set(gca, 'FontName', fontname_main, 'FontSize', fontsize_label);
    
    set(fig, 'Renderer', 'painters', 'PaperUnits', 'centimeters', 'PaperSize', [17.5, 12.5], 'PaperPosition', [0, 0, 17.5, 12.5]);
    drawnow;
    print(fig, fullfile(subdir, '10_crosssection_x0.pdf'), '-dpdf', '-r300', '-fillpage');
    print(fig, fullfile(subdir, '10_crosssection_x0.eps'), '-depsc2', '-r300');
    print(fig, fullfile(subdir, '10_crosssection_x0.png'), '-dpng', '-r300');
    close(fig);
    fprintf('  → 10_crosssection_x0\n');
    
    fprintf('\n✓ All individual subplots saved to: %s\n', subdir);
end