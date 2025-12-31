function u = poisson_fft3(f, hx, hy, hz, niu)
% Solve (-Δ_h + 1/niu) u = f/niu
% on (x1,x2)x(y1,y2)x(z1,z2), zero Dirichlet
% f: nx x ny x nz (interior)

[nx, ny, nz] = size(f);

% ---------- forward DST (FFT-based) ----------
fh = dst3(f / niu);

% ---------- eigenvalues ----------
kx = reshape(1:nx, [], 1, 1);
ky = reshape(1:ny, 1, [], 1);
kz = reshape(1:nz, 1, 1, []);

lambda = ...
    2*(1 - cos(pi*kx/(nx+1))) / hx^2 + ...
    2*(1 - cos(pi*ky/(ny+1))) / hy^2 + ...
    2*(1 - cos(pi*kz/(nz+1))) / hz^2 + ...
    1/niu;

% ---------- spectral solve ----------
uh = fh ./ lambda;

% ---------- inverse DST ----------
u = idst3(uh);
end

% function u = poisson_fft3(f, h, niu)
% % Solve (-Δ_h + 1/niu) u = f/niu
% % (I-niu \Delta) u =f;
% % zero Dirichlet boundary
% % f: nx x ny x nz (interior only)
% 
% [nx, ny, nz] = size(f);
% 
% % ---------- forward DST via odd extension ----------
% fh = dst3(f / niu);
% 
% % ---------- eigenvalues ----------
% kx = reshape(1:nx, [], 1, 1);
% ky = reshape(1:ny, 1, [], 1);
% kz = reshape(1:nz, 1, 1, []);
% 
% lambda = ...
%     2*(1 - cos(pi*kx/(nx+1))) / h^2 + ...
%     2*(1 - cos(pi*ky/(ny+1))) / h^2 + ...
%     2*(1 - cos(pi*kz/(nz+1))) / h^2 + ...
%     1/niu;
% 
% % ---------- spectral solve ----------
% uh = fh ./ lambda;
% 
% % ---------- inverse DST ----------
% u = idst3(uh);
% 
% end
