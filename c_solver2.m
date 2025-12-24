function u = c_solver2(c_boundary, f, dx, dy, niu)
% Solve the elliptic PDE:
%     (I - niu * Laplace) u = f   in Omega
% with Dirichlet boundary condition:
%     u = c_boundary              on ∂Omega
%
% The equation is discretized using a 5-point finite difference scheme
% on a uniform Cartesian grid with possibly different spacing in x and y.
%
% Grid convention:
%   - i = row index, corresponds to y-direction, spacing dy
%   - j = column index, corresponds to x-direction, spacing dx
%
% Inputs:
%   c_boundary : boundary values (N-by-M matrix, only boundary used)
%   f          : right-hand side (N-by-M matrix)
%   dx         : grid spacing in x-direction (column direction)
%   dy         : grid spacing in y-direction (row direction)
%                (if empty, defaults to dx for backward compatibility)
%   niu        : regularization / diffusion parameter
%
% Output:
%   u          : solution (N-by-M matrix)

% Default: dy = dx (square grid) if dy is empty
if isempty(dy)
    dy = dx;
end

[N, M] = size(f); % N = rows (y-direction), M = columns (x-direction)

f = f(2:N-1, 2:M-1) / niu;
% Extract interior values of the right-hand side
% Scaling by 1/niu comes from rewriting:
%     (I - niu * Laplace) u = f
% as:
%     (1/niu * I - Laplace) u = f / niu

% Add boundary contributions to RHS
% Left boundary (j=1): contributes c_boundary(:,1) / dx^2
f(:, 1) = f(:, 1) + c_boundary(2:N-1, 1) / dx^2;
% Right boundary (j=M): contributes c_boundary(:,M) / dx^2
f(:, end) = f(:, end) + c_boundary(2:N-1, end) / dx^2;
% Top boundary (i=1): contributes c_boundary(1,:) / dy^2
f(1, :) = f(1, :) + c_boundary(1, 2:M-1) / dy^2;
% Bottom boundary (i=N): contributes c_boundary(N,:) / dy^2
f(end, :) = f(end, :) + c_boundary(end, 2:M-1) / dy^2;

% Build sparse matrix A for interior unknowns
% Discretized equation at interior point (i,j):
%   (1/niu + 2/dx^2 + 2/dy^2) u_{i,j}
%   - u_{i,j-1}/dx^2 - u_{i,j+1}/dx^2
%   - u_{i-1,j}/dy^2 - u_{i+1,j}/dy^2  = f_{i,j}/niu

% C: tridiagonal matrix for row-direction (i) coupling within same column j
%    Size: (N-2) x (N-2)
%    Diagonal: 2/dy^2 + 2/dx^2 + 1/niu
%    Sub/super-diagonals: -1/dy^2
e = ones(N-2, 1);
diag_coeff = 2/dx^2 + 2/dy^2 + 1/niu;
C = spdiags([-e/dy^2, e*diag_coeff, -e/dy^2], [-1, 0, 1], N-2, N-2);

% D: coupling between adjacent columns j (x-direction)
%    Each row-index couples only to itself across neighboring columns
D = -1/dx^2 * speye(N-2);

% Assemble full 2D operator using Kronecker products
% Unknowns ordered column-wise: u(2:N-1, 2), u(2:N-1, 3), ..., u(2:N-1, M-1)
% MATLAB's (:) operator is column-major, so f(:) stacks columns
e = ones(M-2, 1);
A = kron(speye(M-2), C) + kron(spdiags([e e], [-1 1], M-2, M-2), D);

% Solve the linear system
% No transpose needed: f(:) already in column-major order matching A
u_vec = A \ f(:);

% Assemble solution with boundary values
% reshape with (N-2, M-2) matches the column-major ordering
u = zeros(N, M);
u(2:N-1, 2:M-1) = reshape(u_vec, N-2, M-2);
u(:, [1, end]) = c_boundary(:, [1, end]);
u([1, end], :) = c_boundary([1, end], :);

end