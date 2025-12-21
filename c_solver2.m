function u = c_solver2(c_boundary, f, dx, ~, niu)
% Solve the elliptic PDE:
%     (I - niu * Laplace) u = f   in Omega
% with Dirichlet boundary condition:
%     u = c_boundary              on ∂Omega
%
% The equation is discretized using a 5-point finite difference scheme
% on a uniform Cartesian grid.
%
% dx  : mesh size (assume dx = dy)
% niu : regularization / diffusion parameter

[N, M] = size(f); % Get grid size of the computational domain
h = dx;

f = f(2:N-1, 2:M-1) / niu;
% Extract interior values of the right-hand side
% Scaling by 1/niu comes from rewriting:
%     (I - niu * Laplace) u = f
% as:
%     (1/niu * I - Laplace) u = f / niu

% boundary value fixed 
f(:, 1) = f(:, 1)+c_boundary(2:end-1, 1) / h^2;
% Add contribution from the left boundary (x-direction)
f(:,end)=f(:,end)+c_boundary(2:end-1, end) / h^2;
% Add contribution from the right boundary (x-direction)
f(1, :) = f(1, :)+c_boundary(1, 2:end-1) / h^2;
% Add contribution from the bottom boundary (y-direction)
f(end,:)=f(end,:)+c_boundary(end, 2:end-1) / h^2;
% Add contribution from the top boundary (y-direction)

e=ones(N-2,1);
C = spdiags([-e/h^2, e*(4/h^2 + 1/niu), -e/h^2], [-1, 0, 1], N-2, N-2);
% C is a tridiagonal matrix representing the 1D finite difference operator
% in the x-direction for interior grid points.
%
% For a fixed y-index j, C corresponds to the discretization of:
%
%   (1/niu) * u_{i,j}
%   - (u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}) / h^2
%
% The diagonal entries:
%   (4/h^2 + 1/niu)
% arise from combining the identity operator and the Laplacian.
%
% The sub- and super-diagonals:
%   -1/h^2
% correspond to nearest-neighbor coupling in the x-direction.

D=-1/h^2*eye(N-2);
% D represents the coupling between adjacent rows in the y-direction.
%
% It corresponds to the finite difference terms:
%   - u_{i,j-1} / h^2   and   - u_{i,j+1} / h^2
%
% Note that D is diagonal because each x-index couples only
% to itself across neighboring y-layers.

e=ones(M-2,1);
A=kron(eye(M-2),C)+kron(spdiags([e e],[-1 1],M-2,M-2),D);
% Assemble the 2D finite difference operator A using Kronecker products.
%
% The full operator corresponds to the 5-point stencil:
%
%   (4/h^2 + 1/niu) * u_{i,j}
%   - (u_{i-1,j} + u_{i+1,j}) / h^2
%   - (u_{i,j-1} + u_{i,j+1}) / h^2
%
% The Kronecker structure reflects the ordering of unknowns:
%   - Interior unknowns are stacked column-wise (x varies fastest).
%
% Term 1: kron(I_y, C)
%   - Applies the x-direction operator C independently
%     on each horizontal row (fixed y).
%
% Term 2: kron(T_y, D)
%   - Couples neighboring rows in the y-direction.
%   - T_y is a tridiagonal matrix with ones on the ±1 diagonals,
%     encoding adjacency in y.
%
% Together, these two terms form the standard 2D Laplacian
% plus identity operator in sparse matrix form.

f=f'; u=zeros(N, M);
u(2:N-1,2:M-1)=reshape(A\f(:),M-2,N-2)';
u(:, [1, end]) = c_boundary(:, [1, end]);
u([1, end], :) = c_boundary([1, end], :);

% Summary:
% This routine solves a 2D elliptic equation of the form
% (I - niu * Laplace) u = f using finite differences,
% sparse Kronecker-structured matrices, and Dirichlet boundary conditions.