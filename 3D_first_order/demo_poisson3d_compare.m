%% demo_compare_poisson3d_original_vs_xyz.m
clc; clear; close all;

%% ====== sizes must match your original design ======
I = 25; J = 25; K = 25;     % full grid size (including boundary)
nx = I-2; ny = J-2; nz = K-2;   % interior size for f

%% ====== grid spacing ======
h  = 0.08333;        % original uses h
dx = h; dy = h; dz = h;   % make them identical for fair comparison
niu = 1.0;

%% ====== build RHS f on INTERIOR only ======
f = zeros(nx,ny,nz);

%% ====== build Dirichlet bc on FULL grid (I,J,K) ======
% 这里 bc_full 就扮演你代码里的 "bc"，尺寸必须是 (nx+2,ny+2,nz+2) = (I,J,K)
bc_full = zeros(I,J,K);

% 给一个非平凡边界（避免全零太“巧合”）
[xg,yg,zg] = ndgrid(linspace(-1,1,I), linspace(-1,1,J), linspace(-1,1,K));
bc_val = 0.3*sin(pi*xg) + 0.2*cos(pi*yg) + 0.1*zg;

% 只填边界六个面（内部保持 0 不影响，因为 Dirichlet 只需要边界）
bc_full(1,:,:)   = bc_val(1,:,:);
bc_full(end,:,:) = bc_val(end,:,:);
bc_full(:,1,:)   = bc_val(:,1,:);
bc_full(:,end,:) = bc_val(:,end,:);
bc_full(:,:,1)   = bc_val(:,:,1);
bc_full(:,:,end) = bc_val(:,:,end);

%% ====== call your ORIGINAL solver (h version) ======
tic;
u1 = poisson3d_fd_fullbc(f, bc_full, h, niu);
t1 = toc;

%% ====== call your XYZ solver (dx,dy,dz version) ======
tic;
u2 = poisson3d_fd_fullbc_xyz(f, bc_full, dx, dy, dz, niu);
t2 = toc;

%% ====== compare ======
d = u1 - u2;
fprintf('\n=== Compare poisson3d_fd_fullbc vs poisson3d_fd_fullbc_xyz ===\n');
fprintf('I,J,K = (%d,%d,%d), dx=dy=dz=h=%.5g, niu=%.5g\n', I,J,K,h,niu);
fprintf('time(original) = %.4f s\n', t1);
fprintf('time(xyz)      = %.4f s\n', t2);
fprintf('max|diff|      = %.3e\n', max(abs(d(:))));
fprintf('relL2(diff)    = %.3e\n', norm(d(:))/max(norm(u1(:)),1e-14));

%% ====== optional: check discrete residual on interior for both ======
% residual r = (-niu*Δ + I)u - f   (only on interior)
u1I = u1(2:I-1,2:J-1,2:K-1);
u2I = u2(2:I-1,2:J-1,2:K-1);

lap1 = (u1(3:I,2:J-1,2:K-1) - 2*u1I + u1(1:I-2,2:J-1,2:K-1))/dx^2 ...
     + (u1(2:I-1,3:J,2:K-1) - 2*u1I + u1(2:I-1,1:J-2,2:K-1))/dy^2 ...
     + (u1(2:I-1,2:J-1,3:K) - 2*u1I + u1(2:I-1,2:J-1,1:K-2))/dz^2;
res1 = (-niu*lap1 + u1I) - f;

lap2 = (u2(3:I,2:J-1,2:K-1) - 2*u2I + u2(1:I-2,2:J-1,2:K-1))/dx^2 ...
     + (u2(2:I-1,3:J,2:K-1) - 2*u2I + u2(2:I-1,1:J-2,2:K-1))/dy^2 ...
     + (u2(2:I-1,2:J-1,3:K) - 2*u2I + u2(2:I-1,2:J-1,1:K-2))/dz^2;
res2 = (-niu*lap2 + u2I) - f;

fprintf('\n=== Interior residual check ===\n');
fprintf('||res(original)||_inf = %.3e\n', norm(res1(:),inf));
fprintf('||res(xyz)||_inf      = %.3e\n', norm(res2(:),inf));
fprintf('max|res1-res2|         = %.3e\n', max(abs(res1(:)-res2(:))));

%% ====== quick visualize a mid slice ======
midk = round(K/2);
figure('Position',[120 120 1300 420]);

subplot(1,3,1);
imagesc(u1(:,:,midk)'); axis xy equal tight; colorbar; title('u1 original (mid z)');

subplot(1,3,2);
imagesc(u2(:,:,midk)'); axis xy equal tight; colorbar; title('u2 xyz (mid z)');

subplot(1,3,3);
imagesc(d(:,:,midk)'); axis xy equal tight; colorbar; title('diff (mid z)');