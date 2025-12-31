clear; clc; close all;

% ---------- 1) 生成一个简单 3D 体数据：球体 + 平滑背景 ----------
m = 64; n = 64; p = 48;
[x,y,z] = ndgrid(linspace(-1,1,m), linspace(-1,1,n), linspace(-1,1,p));

r = sqrt(x.^2 + y.^2 + z.^2);
u_true = 0.2 + 0.8*(r <= 0.45);          % 一个硬边界球
u_true = u_true + 0.1*exp(-6*(r-0.7).^2);% 外圈平滑 bump

% ---------- 2) 加噪声 ----------
sigma = 0.15;
f = u_true + sigma*randn(size(u_true));

% ---------- 3) 初始化对偶变量 ----------
w1 = zeros(size(f));
w2 = zeros(size(f));
w3 = zeros(size(f));

% ---------- 4) 参数（先用保守点的） ----------
lbd = 0.1;          % fidelity 参数：大 -> 更贴近 f，小 -> 更平滑
NIT = 400;
GapTol = 0;

% ---------- 5) 运行 ----------
[u, w1,w2,w3, Energy, Dgap, TimeCost, itr] = TV_PDHD_3D(w1,w2,w3,f,1/lbd,NIT,GapTol);

fprintf('\nDone. itr=%d, Dgap=%.3e, Time=%.2fs\n', itr, Dgap, TimeCost);

% ---------- 6) 可视化：看中间切片（带 colorbar，统一色标） ----------
ks = round(p/2);

% 统一色标：只用 u_true 和 u 来定范围（对应你说的 c_ex 与 c_so）
clim_ex_so = [min([u_true(:); u(:)]), max([u_true(:); u(:)])];

figure;

subplot(1,3,1);
imagesc(u_true(:,:,ks));
axis image off;
title('u\_true (mid z)');
% colormap(gray);
caxis(clim_ex_so);
colorbar;

subplot(1,3,2);
imagesc(f(:,:,ks));
axis image off;
title('noisy f (mid z)');
% colormap(gray);
caxis(clim_ex_so);   % 想让 noisy 单独自适应就把这一行删掉
colorbar;

subplot(1,3,3);
imagesc(u(:,:,ks));
axis image off;
title('denoised u (mid z)');
% colormap(gray);
caxis(clim_ex_so);
colorbar;

% ---------- 7) 简单指标 ----------
mse_noisy = mean((f(:) - u_true(:)).^2);
mse_deno  = mean((u(:) - u_true(:)).^2);
fprintf('MSE noisy = %.4e, MSE denoised = %.4e\n', mse_noisy, mse_deno);