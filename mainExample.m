% 1st order
clc; clear; close all
tic
format long

%% parameter settings
tol = 0.001;
kmax = 2000;
N = 129;
x = linspace(-1, 1, N);
z = linspace(-1, 1, N);
[X, Z] = meshgrid(x, z);

%% examples
% c_exact = 3 - 2.5 * exp(-X.^2 / 2);
% c_exact = ones(N);
% c_exact = 3 - (1/2)*exp(-(X.^2 + (Z-0.5).^2)/0.5.^2) - exp(-(X.^2 + (Z-1.25).^2)/0.5.^2);
% load c_data/example_4
load /Users/zhukai/Documents/code/MATLAB/Eikonal equations/Marmousi4Yuxiao/marmousi_smooth.txt


% piece wise solution
% base level
c_exact = ones(N);     % 以 1 为底面（你也可以用 3）

% patch 1: 中央矩形（正扰动）
mask1 = (X > -0.4 & X < 0.4 & Z > -0.2 & Z < 0.2);
c_exact(mask1) = c_exact(mask1) + 1.5;

% patch 2: 左上角矩形（负扰动）
mask2 = (X > -0.9 & X < -0.5 & Z > 0.4 & Z < 0.8);
c_exact(mask2) = c_exact(mask2) - 0.8;

% patch 3: 右下角矩形（正扰动）
mask3 = (X > 0.3 & X < 0.8 & Z > -0.8 & Z < -0.4);
c_exact(mask3) = c_exact(mask3) + 1.0;
c_exact = max(c_exact, 0.5);% 保证正性（重要）

%% source point
% fixed_pt_list = [0, 65, 45];
fixed_pt_list = [];
for m = 1:4
    for n = 1:4
        fixed_pt_list = [fixed_pt_list; 0, m*25, n*25];
        % Add data row by row
    end
end
% for m = 1:10
%     for n = 1:10
%         fixed_pt_list = [fixed_pt_list; 0, m*12, n*12];
%     end
% end    


%% iteration
I = N;
J = N;
dx = (x(end)-x(1)) / (I-1); dy = (z(end) - z(1)) / (J-1);

niu = 1;
% (I - niu laplace) c0 = 0, with exact Dirichlet boundary. 
c0 = c_solver2(c_exact, zeros(I, J), dx, dy, niu);
c = c0;

energy = 1e9;
% alpha_f = 1e-4; alpha_0 = 0.1;
for k = 1: kmax
    
    energy_p = 0;
    cstar = 0;
    parfor p_num = 1:size(fixed_pt_list, 1)
        T = TravelTime_solver(c, fixed_pt_list(p_num, :), dx, dy, I, J);
        T_star = TravelTime_solver(c_exact, fixed_pt_list(p_num, :), dx, dy, I, J);
        energy_p = energy_p + EnergyFun(T, T_star, dx, dy);
        cstar = cstar + cStarSolver(T, T_star, dx, I, J, c);
    end
    
    if energy_p < tol
        break
    end

    energy = [energy, energy_p];
    if mod(k, 10) == 0
       disp(k)
       disp(energy(k+1))
    end

        alpha = 0.01;
%     alpha = alpha_f + 0.5*(alpha_0 - alpha_f) * (1 + cos(pi*k / kmax));
    c = c + alpha * cstar;
   
end

%% results and visualization
norm(c-c_exact)*dx*dx

figure
subplot(2, 2, 1)
mesh(x, z, c_exact)
xlabel x
ylabel z
title('c_{exact}', 'FontWeight','bold')
subplot(2, 2, 2)
mesh(x, z, c0)
xlabel x
ylabel z
title('c_0', 'FontWeight','bold')
subplot(2, 2, 3)
mesh(x, z, c)
xlabel x
ylabel z
title('c_{solution}', 'FontWeight','bold')
subplot(2, 2, 4)
mesh(x, z, (c-c_exact)./c_exact)
title('Relative error')

figure
subplot(2, 2, 1)
plot(energy(2:end), 'linewidth', 1.5)
title('Energy')

subplot(2, 2, 2)
plot(z, c0(60, :), '-.', 'linewidth', 1.5)
hold on
plot(z, c(60, :), '--', 'linewidth', 1.5)
hold on
plot(z, c_exact(60, :), 'linewidth', 1.5)
legend('c_0', 'c_{numerical}', 'c_{exact}')
title('Cross-sections of the solutions: x = 0', 'FontWeight','bold')
set(gca, 'FontName','Times New Roman', 'FontSize',16, 'FontWeight','bold');

subplot(2, 2, 3)
plot(x, c0(:, 60), '-.', 'linewidth', 1.5)
hold on
plot(x, c(:, 60), '--', 'linewidth', 1.5)
hold on
plot(x, c_exact(:, 60), 'linewidth', 1.5)
legend('c_0', 'c_{numerical}', 'c_{exact}')
title('Cross-sections of the solutions: z = 0', 'FontWeight','bold')
set(gca, 'FontName','Times New Roman', 'FontSize',16, 'FontWeight','bold');


toc